
import os
import glob
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA_DIR = "./input"

train_candidates = glob.glob(os.path.join(DATA_DIR, "**", "train.csv"), recursive=True)
test_candidates = glob.glob(os.path.join(DATA_DIR, "**", "test.csv"), recursive=True)

train_path = sorted(train_candidates)[0]
test_path = sorted(test_candidates)[0]

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["Province_State"] = train["Province_State"].fillna("None")
test["Province_State"] = test["Province_State"].fillna("None")
train["Country_Region"] = train["Country_Region"].fillna("None")
test["Country_Region"] = test["Country_Region"].fillna("None")

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

train["geo"] = train["Country_Region"] + "_" + train["Province_State"]
test["geo"] = test["Country_Region"] + "_" + test["Province_State"]

all_df = pd.concat([train, test], sort=False).reset_index(drop=True)
all_df = all_df.sort_values(["geo", "Date"]).reset_index(drop=True)

all_df["day"] = (all_df["Date"] - all_df["Date"].min()).dt.days
all_df["month"] = all_df["Date"].dt.month
all_df["week"] = all_df["Date"].dt.isocalendar().week.astype(int)
all_df["dow"] = all_df["Date"].dt.dayofweek
all_df["geo_id"] = all_df["geo"].astype("category").cat.codes

lag_list = [1, 2, 3, 7, 14]
targets = ["ConfirmedCases", "Fatalities"]

for t in targets:
    all_df[f"log_{t}"] = np.log1p(all_df[t])
    for lag in lag_list:
        all_df[f"{t}_lag{lag}"] = all_df.groupby("geo")[f"log_{t}"].shift(lag)

feature_cols = ["day", "month", "week", "dow", "geo_id"] + [c for c in all_df.columns if "lag" in c]

train_fe = all_df[all_df["ForecastId"].isna()].copy()
test_fe = all_df[all_df["ForecastId"].notna()].copy()

train_fe[feature_cols] = train_fe[feature_cols].fillna(0)
test_fe[feature_cols] = test_fe[feature_cols].fillna(0)

max_train_date = train_fe["Date"].max()
val_days = min(14, max(7, train_fe["Date"].nunique() // 6))
val_start_date = max_train_date - pd.Timedelta(days=val_days - 1)

tr_idx = train_fe["Date"] < val_start_date
va_idx = train_fe["Date"] >= val_start_date

X_train = train_fe.loc[tr_idx, feature_cols].copy()
X_valid = train_fe.loc[va_idx, feature_cols].copy()

valid_pred_df = train_fe.loc[va_idx, ["geo", "Date", "ConfirmedCases", "Fatalities"]].copy()

def rmsle_multi(y_true_cases, y_pred_cases, y_true_fatal, y_pred_fatal):
    y_true_cases = np.maximum(0, np.asarray(y_true_cases))
    y_pred_cases = np.maximum(0, np.asarray(y_pred_cases))
    y_true_fatal = np.maximum(0, np.asarray(y_true_fatal))
    y_pred_fatal = np.maximum(0, np.asarray(y_pred_fatal))
    err_cases = (np.log1p(y_pred_cases) - np.log1p(y_true_cases)) ** 2
    err_fatal = (np.log1p(y_pred_fatal) - np.log1p(y_true_fatal)) ** 2
    return np.sqrt(np.mean(np.concatenate([err_cases, err_fatal])))

preds_valid = {}
preds_test = {}

for target in targets:
    y_train = np.log1p(train_fe.loc[tr_idx, target].values)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)

    p_valid = np.expm1(model.predict(X_valid))
    p_test = np.expm1(model.predict(test_fe[feature_cols]))

    valid_pred_df[f"pred_{target}"] = np.maximum(0, p_valid)
    test_fe[f"pred_{target}"] = np.maximum(0, p_test)

for target in targets:
    valid_pred_df = valid_pred_df.sort_values(["geo", "Date"]).reset_index(drop=True)
    valid_pred_df[f"pred_{target}"] = valid_pred_df.groupby("geo")[f"pred_{target}"].cummax()

    test_fe = test_fe.sort_values(["geo", "Date"]).reset_index(drop=True)
    test_fe[f"pred_{target}"] = test_fe.groupby("geo")[f"pred_{target}"].cummax()

final_validation_score = rmsle_multi(
    valid_pred_df["ConfirmedCases"].values,
    valid_pred_df["pred_ConfirmedCases"].values,
    valid_pred_df["Fatalities"].values,
    valid_pred_df["pred_Fatalities"].values
)

submission = pd.DataFrame({
    "ForecastId": test_fe["ForecastId"].astype(int),
    "ConfirmedCases": test_fe["pred_ConfirmedCases"].values,
    "Fatalities": test_fe["pred_Fatalities"].values
})

submission = submission.sort_values("ForecastId").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
