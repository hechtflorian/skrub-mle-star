
import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

INPUT_DIR = "./input"


def find_file(possible_names):
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f in possible_names:
                return os.path.join(root, f)
    raise FileNotFoundError(f"Could not find any of: {possible_names}")


train_path = find_file(["train.csv"])
test_path = find_file(["test.csv"])

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


def engineer_features(df):
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
        df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
        df = df.drop(columns=["Date"])

    for col in ["Province_State", "Country_Region"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    if "ForecastId" in df.columns:
        df = df.drop(columns=["ForecastId"])

    df = pd.get_dummies(
        df,
        columns=[c for c in ["Province_State", "Country_Region"] if c in df.columns],
        dummy_na=False,
    )

    df.columns = [
        re.sub(r'[^A-Za-z0-9_]+', '_', str(col)).strip('_') or "col"
        for col in df.columns
    ]
    return df


def rmsle(y_true, y_pred):
    y_true = np.maximum(np.asarray(y_true), 0)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def align_features(train_x, pred_x):
    pred_x = pred_x.copy()
    for col in train_x.columns:
        if col not in pred_x.columns:
            pred_x[col] = 0
    for col in pred_x.columns:
        if col not in train_x.columns:
            pred_x = pred_x.drop(columns=[col])
    pred_x = pred_x[train_x.columns]
    return pred_x


def train_and_predict_target(train_part, valid_part, test_df, target_col):
    train_feat = engineer_features(train_part)
    valid_feat = engineer_features(valid_part)
    test_feat = engineer_features(test_df)

    X_train = train_feat.drop(columns=["ConfirmedCases", "Fatalities"], errors="ignore")
    y_train = np.log1p(train_part[target_col].values)

    X_valid = valid_feat.drop(columns=["ConfirmedCases", "Fatalities"], errors="ignore")
    X_test = test_feat.drop(columns=["ConfirmedCases", "Fatalities"], errors="ignore")

    X_valid = align_features(X_train, X_valid)
    X_test = align_features(X_train, X_test)

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    pred_valid_log = model.predict(X_valid)
    pred_test_log = model.predict(X_test)

    pred_valid = np.expm1(pred_valid_log)
    pred_test = np.expm1(pred_test_log)

    pred_valid = np.maximum(pred_valid, 0)
    pred_test = np.maximum(pred_test, 0)

    return pred_valid, pred_test


all_dates = pd.to_datetime(train_df["Date"]).sort_values().unique()
split_point = int(len(all_dates) * 0.8)
valid_dates = set(all_dates[split_point:])

train_part = train_df[~pd.to_datetime(train_df["Date"]).isin(valid_dates)].copy()
valid_part = train_df[pd.to_datetime(train_df["Date"]).isin(valid_dates)].copy()

valid_pred_cases, test_pred_cases = train_and_predict_target(
    train_part, valid_part, test_df, "ConfirmedCases"
)
valid_pred_fatal, test_pred_fatal = train_and_predict_target(
    train_part, valid_part, test_df, "Fatalities"
)

score_cases = rmsle(valid_part["ConfirmedCases"].values, valid_pred_cases)
score_fatal = rmsle(valid_part["Fatalities"].values, valid_pred_fatal)
final_validation_score = (score_cases + score_fatal) / 2.0
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    "ForecastId": test_df["ForecastId"],
    "ConfirmedCases": test_pred_cases,
    "Fatalities": test_pred_fatal,
})

submission["ConfirmedCases"] = np.maximum(submission["ConfirmedCases"], 0)
submission["Fatalities"] = np.maximum(submission["Fatalities"], 0)

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
