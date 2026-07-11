
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

try:
    from lightgbm import LGBMRegressor
except Exception:
    from sklearn.ensemble import RandomForestRegressor as LGBMRegressor


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def prepare_features(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["Province_State"] = out["Province_State"].fillna("")
    out["Country_Region"] = out["Country_Region"].fillna("")
    return out


def build_lgbm_predictor(train_part, target_col):
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=[target_col, "ConfirmedCases", "Fatalities", "Id"], errors="ignore").skb.mark_as_X()
    y_train = train_part[target_col].astype(float)
    if target_col in ["ConfirmedCases", "Fatalities"]:
        y_train = np.log1p(y_train)
    y_train = skrub.y(y_train)

    vectorizer = skrub.TableVectorizer()
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    return predictor


def main():
    base_dir = "./input"
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = prepare_features(train_df)
    test_df = prepare_features(test_df)

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    pred_cases = build_lgbm_predictor(train_part, "ConfirmedCases")
    learner_cases = pred_cases.skb.make_learner(fitted=True)
    valid_pred_cases = np.expm1(np.asarray(learner_cases.predict({"data": valid_part})))
    valid_pred_cases = np.clip(valid_pred_cases, 0, None)

    pred_fatal = build_lgbm_predictor(train_part, "Fatalities")
    learner_fatal = pred_fatal.skb.make_learner(fitted=True)
    valid_pred_fatal = np.expm1(np.asarray(learner_fatal.predict({"data": valid_part})))
    valid_pred_fatal = np.clip(valid_pred_fatal, 0, None)

    score_cases = rmsle(valid_part["ConfirmedCases"], valid_pred_cases)
    score_fatal = rmsle(valid_part["Fatalities"], valid_pred_fatal)
    final_validation_score = (score_cases + score_fatal) / 2.0
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
