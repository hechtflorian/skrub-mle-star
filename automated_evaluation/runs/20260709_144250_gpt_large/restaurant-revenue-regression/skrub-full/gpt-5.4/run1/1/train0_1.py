
import math
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
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
    df["open_year"] = open_date.dt.year
    df["open_month"] = open_date.dt.month
    df["open_dayofweek"] = open_date.dt.dayofweek
    df["open_dayofyear"] = open_date.dt.dayofyear
    df = df.drop(columns=["Open Date"])
    return df


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

    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=[target_col, "Id"], errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer_xgb = skrub.TableVectorizer()
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0,
    )
    predictor_xgb = X_train.skb.apply(vectorizer_xgb).skb.apply(xgb_model, y=y_train)
    learner_xgb = predictor_xgb.skb.make_learner(fitted=True)

    vectorizer_cat = skrub.TableVectorizer()
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        random_state=random_state,
        verbose=0,
    )
    predictor_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)
    learner_cat = predictor_cat.skb.make_learner(fitted=True)

    valid_pred_xgb = np.asarray(learner_xgb.predict({"data": valid_part}), dtype=float).ravel()
    valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part}), dtype=float).ravel()

    valid_pred = 0.6 * valid_pred_xgb + 0.4 * valid_pred_cat
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
