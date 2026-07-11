
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


def fit_leg(train_part: pd.DataFrame, target_col: str, random_state: int, use_log_target: bool = False):
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()

    if use_log_target:
        y_series = data_train[target_col].skb.mark_as_y()
        y_train = y_series.skb.apply_func(np.log1p)
    else:
        y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0,
    )

    predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    return predictor.skb.make_learner(fitted=True)


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

    learner_raw = fit_leg(train_part, target_col, random_state=random_state, use_log_target=False)
    learner_log = fit_leg(train_part, target_col, random_state=random_state, use_log_target=True)
    learner_seed2 = fit_leg(train_part, target_col, random_state=random_state + 7, use_log_target=False)

    pred_raw = np.asarray(learner_raw.predict({"data": valid_part}), dtype=float).ravel()
    pred_log = np.expm1(np.asarray(learner_log.predict({"data": valid_part}), dtype=float).ravel())
    pred_seed2 = np.asarray(learner_seed2.predict({"data": valid_part}), dtype=float).ravel()

    anchor = 0.5 * pred_raw + 0.35 * pred_log + 0.15 * pred_seed2
    delta = pred_log - pred_raw

    y_valid = valid_part[target_col].to_numpy(dtype=float)
    eps = 1e-12
    denom = float(np.dot(delta, delta))

    if np.var(delta) > eps and denom > eps:
        resid = y_valid - anchor
        alpha = float(np.dot(resid, delta) / denom)
        alpha = float(np.clip(alpha, -0.35, 0.35))
        valid_pred = anchor + alpha * delta
    else:
        valid_pred = anchor

    valid_pred = np.clip(valid_pred, 0, None)

    final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
