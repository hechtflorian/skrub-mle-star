
import os
import glob
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

DATA_DIR = "./input"
SEED = 42

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
all_df["week"] = all_df["Date"].dt.isocalendar().week.astype(int)
all_df["dow"] = all_df["Date"].dt.dayofweek

targets = ["ConfirmedCases", "Fatalities"]
lags = [1, 7, 14]

for t in targets:
    all_df[f"log_{t}"] = np.log1p(all_df[t])
    for lag in lags:
        all_df[f"{t}_lag{lag}"] = all_df.groupby("geo")[f"log_{t}"].shift(lag)

train_fe = all_df[all_df["ForecastId"].isna()].copy().reset_index(drop=True)
test_fe = all_df[all_df["ForecastId"].notna()].copy().reset_index(drop=True)

features = [
    "Country_Region",
    "Province_State",
    "geo",
    "day",
    "week",
    "dow",
    "ConfirmedCases_lag1",
    "ConfirmedCases_lag7",
    "ConfirmedCases_lag14",
    "Fatalities_lag1",
    "Fatalities_lag7",
    "Fatalities_lag14",
]

cat_features = [0, 1, 2]

train_fe[features] = train_fe[features].fillna(0)
test_fe[features] = test_fe[features].fillna(0)

max_train_date = train_fe["Date"].max()
unique_dates = np.sort(train_fe["Date"].unique())
val_days = min(14, max(7, len(unique_dates) // 6))
val_start_date = max_train_date - pd.Timedelta(days=val_days - 1)

tr_idx = train_fe["Date"] < val_start_date
va_idx = train_fe["Date"] >= val_start_date

train_part = train_fe.loc[tr_idx].copy().reset_index(drop=True)
valid_part = train_fe.loc[va_idx].copy().reset_index(drop=True)

def rmsle_multi(y_true_cases, y_pred_cases, y_true_fatal, y_pred_fatal):
    y_true_cases = np.maximum(0, np.asarray(y_true_cases))
    y_pred_cases = np.maximum(0, np.asarray(y_pred_cases))
    y_true_fatal = np.maximum(0, np.asarray(y_true_fatal))
    y_pred_fatal = np.maximum(0, np.asarray(y_pred_fatal))
    err_cases = (np.log1p(y_pred_cases) - np.log1p(y_true_cases)) ** 2
    err_fatal = (np.log1p(y_pred_fatal) - np.log1p(y_true_fatal)) ** 2
    return np.sqrt(np.mean(np.concatenate([err_cases, err_fatal])))

valid_preds = pd.DataFrame({
    "geo": valid_part["geo"].values,
    "Date": valid_part["Date"].values,
    "ConfirmedCases": valid_part["ConfirmedCases"].values,
    "Fatalities": valid_part["Fatalities"].values
})

for target in targets:
    y_train = np.log1p(train_part[target].values)

    model = CatBoostRegressor(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=SEED
    )

    model.fit(
        train_part[features],
        y_train,
        cat_features=cat_features
    )

    pred_valid = np.expm1(model.predict(valid_part[features]))
    pred_test = np.expm1(model.predict(test_fe[features]))

    valid_preds[f"pred_{target}"] = np.maximum(0, pred_valid)
    test_fe[f"pred_{target}"] = np.maximum(0, pred_test)

valid_preds = valid_preds.sort_values(["geo", "Date"]).reset_index(drop=True)
test_fe = test_fe.sort_values(["geo", "Date"]).reset_index(drop=True)

for target in targets:
    valid_preds[f"pred_{target}"] = valid_preds.groupby("geo")[f"pred_{target}"].cummax()
    test_fe[f"pred_{target}"] = test_fe.groupby("geo")[f"pred_{target}"].cummax()

final_validation_score = rmsle_multi(
    valid_preds["ConfirmedCases"].values,
    valid_preds["pred_ConfirmedCases"].values,
    valid_preds["Fatalities"].values,
    valid_preds["pred_Fatalities"].values
)

full_train = train_fe.copy().reset_index(drop=True)

for target in targets:
    y_full = np.log1p(full_train[target].values)

    model = CatBoostRegressor(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=SEED
    )

    model.fit(
        full_train[features],
        y_full,
        cat_features=cat_features
    )

    pred_test_full = np.expm1(model.predict(test_fe[features]))
    test_fe[f"{target}_final"] = np.maximum(0, pred_test_full)

test_fe = test_fe.sort_values(["geo", "Date"]).reset_index(drop=True)
for target in targets:
    test_fe[f"{target}_final"] = test_fe.groupby("geo")[f"{target}_final"].cummax()

submission = pd.DataFrame({
    "ForecastId": test_fe["ForecastId"].astype(int),
    "ConfirmedCases": test_fe["ConfirmedCases_final"].values,
    "Fatalities": test_fe["Fatalities_final"].values
}).sort_values("ForecastId").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
