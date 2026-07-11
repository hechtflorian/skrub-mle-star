
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor

INPUT_DIR = "./input"

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sub_path = os.path.join(INPUT_DIR, "submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["Province_State"] = train["Province_State"].fillna("NA")
test["Province_State"] = test["Province_State"].fillna("NA")

for df in [train, test]:
    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["dow"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear

for c in ["Province_State", "Country_Region"]:
    le = LabelEncoder()
    le.fit(pd.concat([train[c].astype(str), test[c].astype(str)], axis=0))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))

features = ["Province_State", "Country_Region", "day", "month", "dow", "dayofyear"]

def make_holdout_split(df, date_col="Date", val_days=14):
    max_date = df[date_col].max()
    split_date = max_date - pd.Timedelta(days=val_days)
    tr_idx = df[date_col] <= split_date
    va_idx = df[date_col] > split_date
    return tr_idx, va_idx

tr_idx, va_idx = make_holdout_split(train, val_days=14)
train_part = train.loc[tr_idx].copy()
valid_part = train.loc[va_idx].copy()

valid_pred = pd.DataFrame(index=valid_part.index)
test_pred = pd.DataFrame(index=test.index)

for target in ["ConfirmedCases", "Fatalities"]:
    y_train = np.log1p(train_part[target].values)
    X_train = train_part[features]
    X_valid = valid_part[features]
    X_test = test[features]

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1500,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        random_seed=42,
    )
    model.fit(X_train, y_train)

    valid_pred[target] = np.expm1(model.predict(X_valid)).clip(0)
    test_pred[target] = np.expm1(model.predict(X_test)).clip(0)

# Reasonable evaluation metric for this task: RMSLE on a time-based validation split
y_true_valid = valid_part[["ConfirmedCases", "Fatalities"]].copy()
y_pred_valid = valid_pred[["ConfirmedCases", "Fatalities"]].copy()

rmsle_cases = np.sqrt(mean_squared_log_error(y_true_valid["ConfirmedCases"], y_pred_valid["ConfirmedCases"]))
rmsle_fatal = np.sqrt(mean_squared_log_error(y_true_valid["Fatalities"], y_pred_valid["Fatalities"]))
final_validation_score = float((rmsle_cases + rmsle_fatal) / 2.0)

submission = pd.DataFrame({
    "ForecastId": test["ForecastId"],
    "ConfirmedCases": test_pred["ConfirmedCases"].values,
    "Fatalities": test_pred["Fatalities"].values
})
submission.to_csv(sub_path, index=False)

print(f"Final Validation Performance: {final_validation_score}")
