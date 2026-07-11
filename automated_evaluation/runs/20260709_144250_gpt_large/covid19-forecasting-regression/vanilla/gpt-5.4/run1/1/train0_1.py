
import os
import glob
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
all_df["month"] = all_df["Date"].dt.month
all_df["week"] = all_df["Date"].dt.isocalendar().week.astype(int)
all_df["dow"] = all_df["Date"].dt.dayofweek
all_df["geo_id"] = all_df["geo"].astype("category").cat.codes

targets = ["ConfirmedCases", "Fatalities"]
lag_list = [1, 2, 3, 7, 14]

for t in targets:
    all_df[f"log_{t}"] = np.log1p(all_df[t])
    for lag in lag_list:
        all_df[f"{t}_lag{lag}"] = all_df.groupby("geo")[f"log_{t}"].shift(lag)

train_fe = all_df[all_df["ForecastId"].isna()].copy().reset_index(drop=True)
test_fe = all_df[all_df["ForecastId"].notna()].copy().reset_index(drop=True)

lgb_features = ["day", "month", "week", "dow", "geo_id"] + [c for c in all_df.columns if "lag" in c]
cat_features = [
    "Country_Region",
    "Province_State",
    "geo",
    "day",
    "month",
    "week",
    "dow",
    "ConfirmedCases_lag1",
    "ConfirmedCases_lag7",
    "ConfirmedCases_lag14",
    "Fatalities_lag1",
    "Fatalities_lag7",
    "Fatalities_lag14",
]
cat_feature_indices = [0, 1, 2]

train_fe[lgb_features] = train_fe[lgb_features].fillna(0)
test_fe[lgb_features] = test_fe[lgb_features].fillna(0)
train_fe[cat_features] = train_fe[cat_features].fillna(0)
test_fe[cat_features] = test_fe[cat_features].fillna(0)

max_train_date = train_fe["Date"].max()
unique_dates = np.sort(train_fe["Date"].unique())
val_days = min(14, max(7, len(unique_dates) // 6))
val_start_date = max_train_date - pd.Timedelta(days=val_days - 1)

tr_idx = train_fe["Date"] < val_start_date
va_idx = train_fe["Date"] >= val_start_date

X_train_lgb = train_fe.loc[tr_idx, lgb_features].copy()
X_valid_lgb = train_fe.loc[va_idx, lgb_features].copy()
X_test_lgb = test_fe[lgb_features].copy()

train_part_cat = train_fe.loc[tr_idx, cat_features].copy().reset_index(drop=True)
valid_part_cat = train_fe.loc[va_idx, cat_features].copy().reset_index(drop=True)
test_part_cat = test_fe[cat_features].copy().reset_index(drop=True)

valid_pred_df = train_fe.loc[va_idx, ["geo", "Date", "ConfirmedCases", "Fatalities"]].copy().reset_index(drop=True)

def rmsle_multi(y_true_cases, y_pred_cases, y_true_fatal, y_pred_fatal):
    y_true_cases = np.maximum(0, np.asarray(y_true_cases))
    y_pred_cases = np.maximum(0, np.asarray(y_pred_cases))
    y_true_fatal = np.maximum(0, np.asarray(y_true_fatal))
    y_pred_fatal = np.maximum(0, np.asarray(y_pred_fatal))
    err_cases = (np.log1p(y_pred_cases) - np.log1p(y_true_cases)) ** 2
    err_fatal = (np.log1p(y_pred_fatal) - np.log1p(y_true_fatal)) ** 2
    return np.sqrt(np.mean(np.concatenate([err_cases, err_fatal])))

for target in targets:
    y_train = np.log1p(train_fe.loc[tr_idx, target].values)

    lgb_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED
    )
    lgb_model.fit(X_train_lgb, y_train)

    cat_model = CatBoostRegressor(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=SEED
    )
    cat_model.fit(
        train_part_cat,
        y_train,
        cat_features=cat_feature_indices
    )

    pred_valid_lgb = np.expm1(lgb_model.predict(X_valid_lgb))
    pred_test_lgb = np.expm1(lgb_model.predict(X_test_lgb))

    pred_valid_cat = np.expm1(cat_model.predict(valid_part_cat))
    pred_test_cat = np.expm1(cat_model.predict(test_part_cat))

    pred_valid_ens = 0.5 * np.maximum(0, pred_valid_lgb) + 0.5 * np.maximum(0, pred_valid_cat)
    pred_test_ens = 0.5 * np.maximum(0, pred_test_lgb) + 0.5 * np.maximum(0, pred_test_cat)

    valid_pred_df[f"pred_{target}"] = pred_valid_ens
    test_fe[f"pred_{target}"] = pred_test_ens

valid_pred_df = valid_pred_df.sort_values(["geo", "Date"]).reset_index(drop=True)
test_fe = test_fe.sort_values(["geo", "Date"]).reset_index(drop=True)

for target in targets:
    valid_pred_df[f"pred_{target}"] = valid_pred_df.groupby("geo")[f"pred_{target}"].cummax()
    test_fe[f"pred_{target}"] = test_fe.groupby("geo")[f"pred_{target}"].cummax()

final_validation_score = rmsle_multi(
    valid_pred_df["ConfirmedCases"].values,
    valid_pred_df["pred_ConfirmedCases"].values,
    valid_pred_df["Fatalities"].values,
    valid_pred_df["pred_Fatalities"].values
)

full_train = train_fe.copy().reset_index(drop=True)
full_train_lgb = full_train[lgb_features].copy()
full_train_cat = full_train[cat_features].copy()
full_test_lgb = test_fe[lgb_features].copy()
full_test_cat = test_fe[cat_features].copy()

for target in targets:
    y_full = np.log1p(full_train[target].values)

    lgb_model_full = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED
    )
    lgb_model_full.fit(full_train_lgb, y_full)

    cat_model_full = CatBoostRegressor(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=SEED
    )
    cat_model_full.fit(
        full_train_cat,
        y_full,
        cat_features=cat_feature_indices
    )

    pred_test_full_lgb = np.expm1(lgb_model_full.predict(full_test_lgb))
    pred_test_full_cat = np.expm1(cat_model_full.predict(full_test_cat))

    test_fe[f"{target}_final"] = 0.5 * np.maximum(0, pred_test_full_lgb) + 0.5 * np.maximum(0, pred_test_full_cat)

test_fe = test_fe.sort_values(["geo", "Date"]).reset_index(drop=True)
for target in targets:
    test_fe[f"{target}_final"] = test_fe.groupby("geo")[f"{target}_final"].cummax()

submission = pd.DataFrame({
    "ForecastId": test_fe["ForecastId"].astype(int),
    "ConfirmedCases": test_fe["ConfirmedCases_final"].values,
    "Fatalities": test_fe["Fatalities_final"].values
})

submission = submission.sort_values("ForecastId").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
