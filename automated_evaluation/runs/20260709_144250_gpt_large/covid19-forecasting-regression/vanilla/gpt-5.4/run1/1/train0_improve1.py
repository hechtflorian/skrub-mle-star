
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
eps = 1e-6

all_df = all_df.sort_values(["geo", "Date"]).reset_index(drop=True)

# Base log features and cumulative lag features
for t in targets:
    all_df[f"log_{t}"] = np.log1p(all_df[t])
    for lag in lag_list:
        all_df[f"{t}_lag{lag}"] = all_df.groupby("geo")[f"log_{t}"].shift(lag)

# Increment targets (daily new counts from cumulative series)
for t in targets:
    prev_val = all_df.groupby("geo")[t].shift(1)
    all_df[f"{t}_inc"] = (all_df[t] - prev_val).clip(lower=0)
    all_df[f"log_{t}_inc"] = np.log1p(all_df[f"{t}_inc"])

# Cheap trend / ratio features from recent lags
for t in targets:
    all_df[f"{t}_lag1_minus_lag7"] = all_df[f"{t}_lag1"] - all_df[f"{t}_lag7"]
    all_df[f"{t}_lag1_minus_lag14"] = all_df[f"{t}_lag1"] - all_df[f"{t}_lag14"]
    all_df[f"{t}_lag3_minus_lag7"] = all_df[f"{t}_lag3"] - all_df[f"{t}_lag7"]
    all_df[f"{t}_lag1_div_lag7"] = (all_df[f"{t}_lag1"] + eps) / (all_df[f"{t}_lag7"] + eps)
    all_df[f"{t}_lag1_div_lag14"] = (all_df[f"{t}_lag1"] + eps) / (all_df[f"{t}_lag14"] + eps)
    all_df[f"{t}_lag3_div_lag7"] = (all_df[f"{t}_lag3"] + eps) / (all_df[f"{t}_lag7"] + eps)

# Cross-target ratios/signals, especially useful for Fatalities
all_df["fatal_to_cases_lag1"] = (all_df["Fatalities_lag1"] + eps) / (all_df["ConfirmedCases_lag1"] + eps)
all_df["fatal_to_cases_lag7"] = (all_df["Fatalities_lag7"] + eps) / (all_df["ConfirmedCases_lag7"] + eps)
all_df["fatal_to_cases_lag14"] = (all_df["Fatalities_lag14"] + eps) / (all_df["ConfirmedCases_lag14"] + eps)
all_df["fatal_cases_gap_lag1"] = all_df["ConfirmedCases_lag1"] - all_df["Fatalities_lag1"]
all_df["fatal_cases_gap_lag7"] = all_df["ConfirmedCases_lag7"] - all_df["Fatalities_lag7"]

base_time_cols = ["day", "month", "week", "dow", "geo_id"]
case_feature_cols = base_time_cols + [
    c for c in all_df.columns
    if ("ConfirmedCases_lag" in c) or c.startswith("ConfirmedCases_lag1_") or c.startswith("ConfirmedCases_lag3_")
]
case_feature_cols += [
    "ConfirmedCases_lag1_minus_lag7",
    "ConfirmedCases_lag1_minus_lag14",
    "ConfirmedCases_lag3_minus_lag7",
    "ConfirmedCases_lag1_div_lag7",
    "ConfirmedCases_lag1_div_lag14",
    "ConfirmedCases_lag3_div_lag7",
]

fatal_feature_cols = base_time_cols + [
    c for c in all_df.columns
    if ("Fatalities_lag" in c) or ("ConfirmedCases_lag" in c)
]
fatal_feature_cols += [
    "Fatalities_lag1_minus_lag7",
    "Fatalities_lag1_minus_lag14",
    "Fatalities_lag3_minus_lag7",
    "Fatalities_lag1_div_lag7",
    "Fatalities_lag1_div_lag14",
    "Fatalities_lag3_div_lag7",
    "ConfirmedCases_lag1_minus_lag7",
    "ConfirmedCases_lag1_minus_lag14",
    "ConfirmedCases_lag3_minus_lag7",
    "ConfirmedCases_lag1_div_lag7",
    "ConfirmedCases_lag1_div_lag14",
    "ConfirmedCases_lag3_div_lag7",
    "fatal_to_cases_lag1",
    "fatal_to_cases_lag7",
    "fatal_to_cases_lag14",
    "fatal_cases_gap_lag1",
    "fatal_cases_gap_lag7",
]

case_feature_cols = list(dict.fromkeys(case_feature_cols))
fatal_feature_cols = list(dict.fromkeys(fatal_feature_cols))

train_fe = all_df[all_df["ForecastId"].isna()].copy()
test_fe = all_df[all_df["ForecastId"].notna()].copy()

train_fe[case_feature_cols] = train_fe[case_feature_cols].fillna(0)
train_fe[fatal_feature_cols] = train_fe[fatal_feature_cols].fillna(0)
test_fe[case_feature_cols] = test_fe[case_feature_cols].fillna(0)
test_fe[fatal_feature_cols] = test_fe[fatal_feature_cols].fillna(0)

max_train_date = train_fe["Date"].max()
val_days = min(14, max(7, train_fe["Date"].nunique() // 6))
val_start_date = max_train_date - pd.Timedelta(days=val_days - 1)

tr_idx = train_fe["Date"] < val_start_date
va_idx = train_fe["Date"] >= val_start_date

X_train_cases = train_fe.loc[tr_idx, case_feature_cols].copy()
X_valid_cases = train_fe.loc[va_idx, case_feature_cols].copy()
X_train_fatal = train_fe.loc[tr_idx, fatal_feature_cols].copy()
X_valid_fatal = train_fe.loc[va_idx, fatal_feature_cols].copy()

valid_pred_df = train_fe.loc[va_idx, ["geo", "Date", "ConfirmedCases", "Fatalities"]].copy()

def rmsle_multi(y_true_cases, y_pred_cases, y_true_fatal, y_pred_fatal):
    y_true_cases = np.maximum(0, np.asarray(y_true_cases))
    y_pred_cases = np.maximum(0, np.asarray(y_pred_cases))
    y_true_fatal = np.maximum(0, np.asarray(y_true_fatal))
    y_pred_fatal = np.maximum(0, np.asarray(y_pred_fatal))
    err_cases = (np.log1p(y_pred_cases) - np.log1p(y_true_cases)) ** 2
    err_fatal = (np.log1p(y_pred_fatal) - np.log1p(y_true_fatal)) ** 2
    return np.sqrt(np.mean(np.concatenate([err_cases, err_fatal])))

# Train on daily increments to improve stationarity
y_train_cases = train_fe.loc[tr_idx, "log_ConfirmedCases_inc"].values
y_train_fatal = train_fe.loc[tr_idx, "log_Fatalities_inc"].values

case_model = LGBMRegressor(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

fatal_model = LGBMRegressor(
    n_estimators=900,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

case_model.fit(X_train_cases, y_train_cases)
fatal_model.fit(X_train_fatal, y_train_fatal)

# Validation prediction on increment scale, then reconstruct cumulative totals using actual lag1 baseline
valid_cases_inc = np.maximum(0, np.expm1(case_model.predict(X_valid_cases)))
valid_pred_df["pred_ConfirmedCases"] = (
    train_fe.loc[va_idx, "ConfirmedCases"].groupby(train_fe.loc[va_idx, "geo"]).shift(1).fillna(0).values
    + valid_cases_inc
)

# For fatalities validation, use actual features already present (including confirmed case lags)
valid_fatal_inc = np.maximum(0, np.expm1(fatal_model.predict(X_valid_fatal)))
valid_pred_df["pred_Fatalities"] = (
    train_fe.loc[va_idx, "Fatalities"].groupby(train_fe.loc[va_idx, "geo"]).shift(1).fillna(0).values
    + valid_fatal_inc
)

for target in targets:
    valid_pred_df = valid_pred_df.sort_values(["geo", "Date"]).reset_index(drop=True)
    valid_pred_df[f"pred_{target}"] = valid_pred_df.groupby("geo")[f"pred_{target}"].cummax()

# Recursive forecasting for test
history_cols = ["geo", "Date", "day", "month", "week", "dow", "geo_id", "ForecastId", "ConfirmedCases", "Fatalities"]
hist_df = all_df[history_cols].copy()

# Seed history with known train cumulative values only
hist_df.loc[hist_df["ForecastId"].notna(), ["ConfirmedCases", "Fatalities"]] = np.nan

test_dates = sorted(test_fe["Date"].unique())
geo_groups = hist_df.groupby("geo")

for current_date in test_dates:
    day_rows = test_fe[test_fe["Date"] == current_date].copy()
    if day_rows.empty:
        continue

    pred_cases_list = []
    pred_fatal_list = []

    for idx, row in day_rows.iterrows():
        geo = row["geo"]
        geo_hist = hist_df[(hist_df["geo"] == geo) & (hist_df["Date"] < current_date)].sort_values("Date")

        feat = {
            "day": row["day"],
            "month": row["month"],
            "week": row["week"],
            "dow": row["dow"],
            "geo_id": row["geo_id"],
        }

        for t in targets:
            for lag in lag_list:
                lag_val_series = geo_hist[t].dropna()
                if len(lag_val_series) >= lag:
                    lag_val = lag_val_series.iloc[-lag]
                    feat[f"{t}_lag{lag}"] = np.log1p(max(0, lag_val))
                else:
                    feat[f"{t}_lag{lag}"] = 0.0

            feat[f"{t}_lag1_minus_lag7"] = feat[f"{t}_lag1"] - feat[f"{t}_lag7"]
            feat[f"{t}_lag1_minus_lag14"] = feat[f"{t}_lag1"] - feat[f"{t}_lag14"]
            feat[f"{t}_lag3_minus_lag7"] = feat[f"{t}_lag3"] - feat[f"{t}_lag7"]
            feat[f"{t}_lag1_div_lag7"] = (feat[f"{t}_lag1"] + eps) / (feat[f"{t}_lag7"] + eps)
            feat[f"{t}_lag1_div_lag14"] = (feat[f"{t}_lag1"] + eps) / (feat[f"{t}_lag14"] + eps)
            feat[f"{t}_lag3_div_lag7"] = (feat[f"{t}_lag3"] + eps) / (feat[f"{t}_lag7"] + eps)

        feat["fatal_to_cases_lag1"] = (feat["Fatalities_lag1"] + eps) / (feat["ConfirmedCases_lag1"] + eps)
        feat["fatal_to_cases_lag7"] = (feat["Fatalities_lag7"] + eps) / (feat["ConfirmedCases_lag7"] + eps)
        feat["fatal_to_cases_lag14"] = (feat["Fatalities_lag14"] + eps) / (feat["ConfirmedCases_lag14"] + eps)
        feat["fatal_cases_gap_lag1"] = feat["ConfirmedCases_lag1"] - feat["Fatalities_lag1"]
        feat["fatal_cases_gap_lag7"] = feat["ConfirmedCases_lag7"] - feat["Fatalities_lag7"]

        X_case_row = pd.DataFrame([feat])[case_feature_cols].fillna(0)
        pred_case_inc = max(0.0, np.expm1(case_model.predict(X_case_row)[0]))

        prev_cases = geo_hist["ConfirmedCases"].dropna()
        prev_cases_val = prev_cases.iloc[-1] if len(prev_cases) > 0 else 0.0
        pred_cases = max(prev_cases_val, prev_cases_val + pred_case_inc)

        # Update case-dependent features for fatalities with same-day recursive case prediction not needed directly,
        # but lag features are from history only; fatalities model already benefits from case lags.
        X_fatal_row = pd.DataFrame([feat])[fatal_feature_cols].fillna(0)
        pred_fatal_inc = max(0.0, np.expm1(fatal_model.predict(X_fatal_row)[0]))

        prev_fatal = geo_hist["Fatalities"].dropna()
        prev_fatal_val = prev_fatal.iloc[-1] if len(prev_fatal) > 0 else 0.0
        pred_fatal = max(prev_fatal_val, prev_fatal_val + pred_fatal_inc)
        pred_fatal = min(pred_fatal, pred_cases)

        pred_cases_list.append((idx, pred_cases))
        pred_fatal_list.append((idx, pred_fatal))

    for idx, pred_cases in pred_cases_list:
        test_fe.loc[idx, "pred_ConfirmedCases"] = pred_cases
        hist_df.loc[
            (hist_df["geo"] == test_fe.loc[idx, "geo"]) & (hist_df["Date"] == current_date),
            "ConfirmedCases"
        ] = pred_cases

    for idx, pred_fatal in pred_fatal_list:
        test_fe.loc[idx, "pred_Fatalities"] = pred_fatal
        hist_df.loc[
            (hist_df["geo"] == test_fe.loc[idx, "geo"]) & (hist_df["Date"] == current_date),
            "Fatalities"
        ] = pred_fatal

for target in targets:
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
