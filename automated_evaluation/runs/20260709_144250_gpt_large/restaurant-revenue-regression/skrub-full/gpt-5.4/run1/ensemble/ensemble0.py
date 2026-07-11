
import math
import numpy as np
import pandas as pd
import skrub
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    open_date = pd.to_datetime(df["Open Date"], format="%m/%d/%Y")
    ref_date = pd.Timestamp("2015-01-01")
    age_days = (ref_date - open_date).dt.days
    df["restaurant_age_days"] = age_days
    df["restaurant_age_log"] = age_days.clip(lower=1).map(math.log)
    df = df.drop(columns=["Open Date"])
    return df


def build_model(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0,
    )


def fit_leg(train_leg_df: pd.DataFrame, target_col: str, random_state: int):
    data_train = skrub.var("data", train_leg_df)
    X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = build_model(random_state=random_state)

    predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = predictor.skb.make_learner(fitted=True)
    return learner


def main():
    train_path = "./input/train.csv"
    train_df = pd.read_csv(train_path)

    target_col = "revenue"
    random_state = 42

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=random_state
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    train_part = add_features(train_part)
    valid_part = add_features(valid_part)

    # Leg 1: original raw-target model
    learner_raw = fit_leg(train_part.copy(), target_col, random_state)
    pred_raw = np.asarray(learner_raw.predict({"data": valid_part}), dtype=float).ravel()

    # Leg 2: same pipeline on log1p target, then invert with expm1
    train_part_log = train_part.copy()
    train_part_log[target_col] = np.log1p(train_part_log[target_col])
    learner_log = fit_leg(train_part_log, target_col, random_state)
    pred_log = np.asarray(learner_log.predict({"data": valid_part}), dtype=float).ravel()
    pred_log = np.expm1(pred_log)

    # Leg 3: same raw-target pipeline on a resampled subset of train_part
    resample_frac = 0.85
    train_part_resampled = train_part.sample(
        frac=resample_frac, replace=False, random_state=random_state + 1
    ).copy()
    learner_resampled = fit_leg(train_part_resampled, target_col, random_state + 1)
    pred_resampled = np.asarray(
        learner_resampled.predict({"data": valid_part}), dtype=float
    ).ravel()

    valid_pred = 0.45 * pred_raw + 0.40 * pred_log + 0.15 * pred_resampled
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
