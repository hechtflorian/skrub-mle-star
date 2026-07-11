
import os
import sys
import glob
import subprocess
import numpy as np
import pandas as pd

def ensure_package(package_name):
    try:
        __import__(package_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_package("lightgbm")

import lightgbm as lgb
from lightgbm import LGBMRegressor

DATA_DIR = "./input"

train_candidates = glob.glob(os.path.join(DATA_DIR, "**", "train.csv"), recursive=True)
test_candidates = glob.glob(os.path.join(DATA_DIR, "**", "test.csv"), recursive=True)

if len(train_candidates) == 0:
    raise FileNotFoundError("train.csv not found under ./input")
if len(test_candidates) == 0:
    raise FileNotFoundError("test.csv not found under ./input")

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

targets = ["ConfirmedCases", "Fatalities"]

def rmsle_multi(y_true_cases, y_pred_cases, y_true_fatal, y_pred_fatal):
    y_true_cases = np.maximum(0, np.asarray(y_true_cases))
    y_pred_cases = np.maximum(0, np.asarray(y_pred_cases))
    y_true_fatal = np.maximum(0, np.asarray(y_true_fatal))
    y_pred_fatal = np.maximum(0, np.asarray(y_pred_fatal))
    err_cases = (np.log1p(y_pred_cases) - np.log1p(y_true_cases)) ** 2
    err_fatal = (np.log1p(y_pred_fatal) - np.log1p(y_true_fatal)) ** 2
    return np.sqrt(np.mean(np.concatenate([err_cases, err_fatal])))

def post_process_monotonic(df_pred, test_meta):
    out = df_pred.merge(
        test_meta[["ForecastId", "geo", "Date"]],
        on="ForecastId",
        how="left"
    )
    out = out.sort_values(["geo", "Date", "ForecastId"]).reset_index(drop=True)
    out["ConfirmedCases"] = np.maximum(0, out["ConfirmedCases"].values)
    out["Fatalities"] = np.maximum(0, out["Fatalities"].values)
    out["ConfirmedCases"] = out.groupby("geo")["ConfirmedCases"].cummax()
    out["Fatalities"] = out.groupby("geo")["Fatalities"].cummax()
    out["Fatalities"] = np.minimum(out["Fatalities"].values, out["ConfirmedCases"].values)
    return out[["ForecastId", "ConfirmedCases", "Fatalities"]].sort_values("ForecastId").reset_index(drop=True)

# -----------------------------
# Model A: Original cumulative LightGBM solution
# -----------------------------
all_df_a = pd.concat([train, test], sort=False).reset_index(drop=True)
all_df_a = all_df_a.sort_values(["geo", "Date"]).reset_index(drop=True)

all_df_a["day"] = (all_df_a["Date"] - all_df_a["Date"].min()).dt.days
all_df_a["month"] = all_df_a["Date"].dt.month
all_df_a["week"] = all_df_a["Date"].dt.isocalendar().week.astype(int)
all_df_a["dow"] = all_df_a["Date"].dt.dayofweek
all_df_a["geo_id"] = all_df_a["geo"].astype("category").cat.codes

lag_list = [1, 2, 3, 7, 14]

for t in targets:
    all_df_a[f"log_{t}"] = np.log1p(all_df_a[t].fillna(0))
    for lag in lag_list:
        all_df_a[f"{t}_lag{lag}"] = all_df_a.groupby("geo")[f"log_{t}"].shift(lag)

    all_df_a[f"{t}_diff_lag1_2"] = all_df_a[f"{t}_lag1"] - all_df_a[f"{t}_lag2"]
    all_df_a[f"{t}_diff_lag1_3"] = all_df_a[f"{t}_lag1"] - all_df_a[f"{t}_lag3"]
    all_df_a[f"{t}_diff_lag1_7"] = all_df_a[f"{t}_lag1"] - all_df_a[f"{t}_lag7"]
    all_df_a[f"{t}_diff_lag7_14"] = all_df_a[f"{t}_lag7"] - all_df_a[f"{t}_lag14"]

    all_df_a[f"{t}_rollmean_3"] = all_df_a[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3"]].mean(axis=1)
    all_df_a[f"{t}_rollmean_7"] = all_df_a[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3", f"{t}_lag7"]].mean(axis=1)
    all_df_a[f"{t}_rollmax_3"] = all_df_a[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3"]].max(axis=1)
    all_df_a[f"{t}_rollmin_3"] = all_df_a[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3"]].min(axis=1)

feature_cols_a = ["day", "month", "week", "dow", "geo_id"] + [
    c for c in all_df_a.columns if ("lag" in c or "diff_" in c or "roll" in c)
]

train_fe_a = all_df_a[all_df_a["ForecastId"].isna()].copy()
test_fe_a = all_df_a[all_df_a["ForecastId"].notna()].copy()

train_fe_a[feature_cols_a] = train_fe_a[feature_cols_a].fillna(0)
test_fe_a[feature_cols_a] = test_fe_a[feature_cols_a].fillna(0)

max_train_date = train_fe_a["Date"].max()
val_days = min(14, max(7, train_fe_a["Date"].nunique() // 6))
val_start_date = max_train_date - pd.Timedelta(days=val_days - 1)

tr_idx = train_fe_a["Date"] < val_start_date
va_idx = train_fe_a["Date"] >= val_start_date

X_train_a = train_fe_a.loc[tr_idx, feature_cols_a].copy()
X_valid_a = train_fe_a.loc[va_idx, feature_cols_a].copy()

valid_pred_a = train_fe_a.loc[va_idx, ["geo", "Date", "ConfirmedCases", "Fatalities"]].copy()
test_pred_a = test_fe_a[["ForecastId"]].copy()

for target in targets:
    y_train = np.log1p(train_fe_a.loc[tr_idx, target].values)
    y_valid = np.log1p(train_fe_a.loc[va_idx, target].values)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(
        X_train_a,
        y_train,
        eval_set=[(X_valid_a, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    best_iter = model.best_iteration_ if model.best_iteration_ is not None else model.n_estimators
    p_valid = np.expm1(model.predict(X_valid_a, num_iteration=best_iter))
    p_test = np.expm1(model.predict(test_fe_a[feature_cols_a], num_iteration=best_iter))

    valid_pred_a[f"pred_{target}"] = np.maximum(0, p_valid)
    test_pred_a[target] = np.maximum(0, p_test)

score_a = rmsle_multi(
    valid_pred_a["ConfirmedCases"].values,
    valid_pred_a["pred_ConfirmedCases"].values,
    valid_pred_a["Fatalities"].values,
    valid_pred_a["pred_Fatalities"].values
)

submission_a = pd.DataFrame({
    "ForecastId": test_fe_a["ForecastId"].astype(int),
    "ConfirmedCases": test_pred_a["ConfirmedCases"].values,
    "Fatalities": test_pred_a["Fatalities"].values
}).sort_values("ForecastId").reset_index(drop=True)
submission_a.to_csv("submission_a.csv", index=False)

# -----------------------------
# Model B: Increment-based recursive LightGBM solution
# -----------------------------
train_b = train.copy().sort_values(["geo", "Date"]).reset_index(drop=True)
test_b = test.copy().sort_values(["geo", "Date"]).reset_index(drop=True)

train_b["day"] = (train_b["Date"] - train_b["Date"].min()).dt.days
test_b["day"] = (test_b["Date"] - train_b["Date"].min()).dt.days
train_b["month"] = train_b["Date"].dt.month
test_b["month"] = test_b["Date"].dt.month
train_b["week"] = train_b["Date"].dt.isocalendar().week.astype(int)
test_b["week"] = test_b["Date"].dt.isocalendar().week.astype(int)
train_b["dow"] = train_b["Date"].dt.dayofweek
test_b["dow"] = test_b["Date"].dt.dayofweek

geo_categories = pd.Categorical(pd.concat([train_b["geo"], test_b["geo"]], axis=0))
geo_map = {cat: i for i, cat in enumerate(geo_categories.categories)}
train_b["geo_id"] = train_b["geo"].map(geo_map).astype(int)
test_b["geo_id"] = test_b["geo"].map(geo_map).astype(int)

train_b["log_cases"] = np.log1p(train_b["ConfirmedCases"].clip(lower=0))
train_b["log_fatal"] = np.log1p(train_b["Fatalities"].clip(lower=0))

train_b["inc_cases"] = train_b.groupby("geo")["ConfirmedCases"].diff().fillna(train_b["ConfirmedCases"])
train_b["inc_fatal"] = train_b.groupby("geo")["Fatalities"].diff().fillna(train_b["Fatalities"])
train_b["inc_cases"] = train_b["inc_cases"].clip(lower=0)
train_b["inc_fatal"] = train_b["inc_fatal"].clip(lower=0)
train_b["log_inc_cases"] = np.log1p(train_b["inc_cases"])
train_b["log_inc_fatal"] = np.log1p(train_b["inc_fatal"])

for lag in [1, 2, 3, 7, 14]:
    train_b[f"log_cases_lag{lag}"] = train_b.groupby("geo")["log_cases"].shift(lag)
    train_b[f"log_fatal_lag{lag}"] = train_b.groupby("geo")["log_fatal"].shift(lag)
    train_b[f"log_inc_cases_lag{lag}"] = train_b.groupby("geo")["log_inc_cases"].shift(lag)
    train_b[f"log_inc_fatal_lag{lag}"] = train_b.groupby("geo")["log_inc_fatal"].shift(lag)

train_b["cases_growth_1_2"] = train_b["log_cases_lag1"] - train_b["log_cases_lag2"]
train_b["cases_growth_1_7"] = train_b["log_cases_lag1"] - train_b["log_cases_lag7"]
train_b["fatal_growth_1_2"] = train_b["log_fatal_lag1"] - train_b["log_fatal_lag2"]
train_b["fatal_growth_1_7"] = train_b["log_fatal_lag1"] - train_b["log_fatal_lag7"]

train_b["inc_cases_roll3"] = train_b[[f"log_inc_cases_lag1", f"log_inc_cases_lag2", f"log_inc_cases_lag3"]].mean(axis=1)
train_b["inc_cases_roll7"] = train_b[[f"log_inc_cases_lag1", f"log_inc_cases_lag2", f"log_inc_cases_lag3", f"log_inc_cases_lag7"]].mean(axis=1)
train_b["inc_fatal_roll3"] = train_b[[f"log_inc_fatal_lag1", f"log_inc_fatal_lag2", f"log_inc_fatal_lag3"]].mean(axis=1)
train_b["inc_fatal_roll7"] = train_b[[f"log_inc_fatal_lag1", f"log_inc_fatal_lag2", f"log_inc_fatal_lag3", f"log_inc_fatal_lag7"]].mean(axis=1)

feature_cols_b = [
    "day", "month", "week", "dow", "geo_id",
    "log_cases_lag1", "log_cases_lag2", "log_cases_lag3", "log_cases_lag7", "log_cases_lag14",
    "log_fatal_lag1", "log_fatal_lag2", "log_fatal_lag3", "log_fatal_lag7", "log_fatal_lag14",
    "log_inc_cases_lag1", "log_inc_cases_lag2", "log_inc_cases_lag3", "log_inc_cases_lag7", "log_inc_cases_lag14",
    "log_inc_fatal_lag1", "log_inc_fatal_lag2", "log_inc_fatal_lag3", "log_inc_fatal_lag7", "log_inc_fatal_lag14",
    "cases_growth_1_2", "cases_growth_1_7", "fatal_growth_1_2", "fatal_growth_1_7",
    "inc_cases_roll3", "inc_cases_roll7", "inc_fatal_roll3", "inc_fatal_roll7"
]

train_b[feature_cols_b] = train_b[feature_cols_b].fillna(0)
test_b[feature_cols_b[:5]] = test_b[feature_cols_b[:5]].fillna(0)

tr_idx_b = train_b["Date"] < val_start_date
va_idx_b = train_b["Date"] >= val_start_date

X_train_b = train_b.loc[tr_idx_b, feature_cols_b].copy()
X_valid_b = train_b.loc[va_idx_b, feature_cols_b].copy()

valid_truth_b = train_b.loc[va_idx_b, ["geo", "Date", "ConfirmedCases", "Fatalities"]].copy()

models_b = {}
for target_name in ["log_inc_cases", "log_inc_fatal"]:
    y_train_b = train_b.loc[tr_idx_b, target_name].values
    y_valid_b = train_b.loc[va_idx_b, target_name].values

    model_b = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model_b.fit(
        X_train_b,
        y_train_b,
        eval_set=[(X_valid_b, y_valid_b)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    models_b[target_name] = model_b

valid_sim = train_b.loc[train_b["Date"] < val_start_date, ["geo", "Date", "ConfirmedCases", "Fatalities", "geo_id"]].copy()
valid_dates = sorted(train_b.loc[va_idx_b, "Date"].unique())

valid_preds_collect = []

for current_date in valid_dates:
    date_slice = train_b.loc[train_b["Date"] == current_date, ["geo", "Date", "geo_id", "day", "month", "week", "dow"]].copy()
    pred_rows = []

    for _, row in date_slice.iterrows():
        geo = row["geo"]
        hist = valid_sim[valid_sim["geo"] == geo].sort_values("Date").copy()

        feat = {
            "day": row["day"],
            "month": row["month"],
            "week": row["week"],
            "dow": row["dow"],
            "geo_id": row["geo_id"]
        }

        hist_cases = np.log1p(hist["ConfirmedCases"].clip(lower=0).values)
        hist_fatal = np.log1p(hist["Fatalities"].clip(lower=0).values)
        hist_inc_cases = np.log1p(hist["ConfirmedCases"].diff().fillna(hist["ConfirmedCases"]).clip(lower=0).values)
        hist_inc_fatal = np.log1p(hist["Fatalities"].diff().fillna(hist["Fatalities"]).clip(lower=0).values)

        def get_lag(arr, lag):
            return arr[-lag] if len(arr) >= lag else 0.0

        for lag in [1, 2, 3, 7, 14]:
            feat[f"log_cases_lag{lag}"] = get_lag(hist_cases, lag)
            feat[f"log_fatal_lag{lag}"] = get_lag(hist_fatal, lag)
            feat[f"log_inc_cases_lag{lag}"] = get_lag(hist_inc_cases, lag)
            feat[f"log_inc_fatal_lag{lag}"] = get_lag(hist_inc_fatal, lag)

        feat["cases_growth_1_2"] = feat["log_cases_lag1"] - feat["log_cases_lag2"]
        feat["cases_growth_1_7"] = feat["log_cases_lag1"] - feat["log_cases_lag7"]
        feat["fatal_growth_1_2"] = feat["log_fatal_lag1"] - feat["log_fatal_lag2"]
        feat["fatal_growth_1_7"] = feat["log_fatal_lag1"] - feat["log_fatal_lag7"]

        feat["inc_cases_roll3"] = np.mean([feat["log_inc_cases_lag1"], feat["log_inc_cases_lag2"], feat["log_inc_cases_lag3"]])
        feat["inc_cases_roll7"] = np.mean([feat["log_inc_cases_lag1"], feat["log_inc_cases_lag2"], feat["log_inc_cases_lag3"], feat["log_inc_cases_lag7"]])
        feat["inc_fatal_roll3"] = np.mean([feat["log_inc_fatal_lag1"], feat["log_inc_fatal_lag2"], feat["log_inc_fatal_lag3"]])
        feat["inc_fatal_roll7"] = np.mean([feat["log_inc_fatal_lag1"], feat["log_inc_fatal_lag2"], feat["log_inc_fatal_lag3"], feat["log_inc_fatal_lag7"]])

        X_row = pd.DataFrame([feat], columns=feature_cols_b).fillna(0)

        model_cases_b = models_b["log_inc_cases"]
        model_fatal_b = models_b["log_inc_fatal"]

        best_iter_cases = model_cases_b.best_iteration_ if model_cases_b.best_iteration_ is not None else model_cases_b.n_estimators
        best_iter_fatal = model_fatal_b.best_iteration_ if model_fatal_b.best_iteration_ is not None else model_fatal_b.n_estimators

        pred_log_inc_cases = model_cases_b.predict(X_row, num_iteration=best_iter_cases)[0]
        pred_log_inc_fatal = model_fatal_b.predict(X_row, num_iteration=best_iter_fatal)[0]

        pred_inc_cases = max(0.0, np.expm1(pred_log_inc_cases))
        pred_inc_fatal = max(0.0, np.expm1(pred_log_inc_fatal))

        last_cases = hist["ConfirmedCases"].iloc[-1] if len(hist) > 0 else 0.0
        last_fatal = hist["Fatalities"].iloc[-1] if len(hist) > 0 else 0.0

        pred_cases = max(last_cases, last_cases + pred_inc_cases)
        pred_fatal = max(last_fatal, last_fatal + pred_inc_fatal)
        pred_fatal = min(pred_fatal, pred_cases)

        pred_rows.append({
            "geo": geo,
            "Date": current_date,
            "ConfirmedCases": pred_cases,
            "Fatalities": pred_fatal,
            "geo_id": row["geo_id"]
        })

    pred_day_df = pd.DataFrame(pred_rows)
    valid_preds_collect.append(pred_day_df[["geo", "Date", "ConfirmedCases", "Fatalities"]])
    valid_sim = pd.concat([valid_sim, pred_day_df], ignore_index=True, sort=False)

valid_pred_b = pd.concat(valid_preds_collect, ignore_index=True)
valid_eval_b = valid_truth_b.merge(
    valid_pred_b,
    on=["geo", "Date"],
    how="left",
    suffixes=("_true", "_pred")
)

score_b = rmsle_multi(
    valid_eval_b["ConfirmedCases_true"].values,
    valid_eval_b["ConfirmedCases_pred"].values,
    valid_eval_b["Fatalities_true"].values,
    valid_eval_b["Fatalities_pred"].values
)

history_test = train_b[["geo", "Date", "ConfirmedCases", "Fatalities", "geo_id"]].copy()
test_dates = sorted(test_b["Date"].unique())
test_preds_collect = []

for current_date in test_dates:
    date_slice = test_b.loc[test_b["Date"] == current_date, ["ForecastId", "geo", "Date", "geo_id", "day", "month", "week", "dow"]].copy()
    pred_rows = []

    for _, row in date_slice.iterrows():
        geo = row["geo"]
        hist = history_test[history_test["geo"] == geo].sort_values("Date").copy()

        feat = {
            "day": row["day"],
            "month": row["month"],
            "week": row["week"],
            "dow": row["dow"],
            "geo_id": row["geo_id"]
        }

        hist_cases = np.log1p(hist["ConfirmedCases"].clip(lower=0).values)
        hist_fatal = np.log1p(hist["Fatalities"].clip(lower=0).values)
        hist_inc_cases = np.log1p(hist["ConfirmedCases"].diff().fillna(hist["ConfirmedCases"]).clip(lower=0).values)
        hist_inc_fatal = np.log1p(hist["Fatalities"].diff().fillna(hist["Fatalities"]).clip(lower=0).values)

        def get_lag(arr, lag):
            return arr[-lag] if len(arr) >= lag else 0.0

        for lag in [1, 2, 3, 7, 14]:
            feat[f"log_cases_lag{lag}"] = get_lag(hist_cases, lag)
            feat[f"log_fatal_lag{lag}"] = get_lag(hist_fatal, lag)
            feat[f"log_inc_cases_lag{lag}"] = get_lag(hist_inc_cases, lag)
            feat[f"log_inc_fatal_lag{lag}"] = get_lag(hist_inc_fatal, lag)

        feat["cases_growth_1_2"] = feat["log_cases_lag1"] - feat["log_cases_lag2"]
        feat["cases_growth_1_7"] = feat["log_cases_lag1"] - feat["log_cases_lag7"]
        feat["fatal_growth_1_2"] = feat["log_fatal_lag1"] - feat["log_fatal_lag2"]
        feat["fatal_growth_1_7"] = feat["log_fatal_lag1"] - feat["log_fatal_lag7"]

        feat["inc_cases_roll3"] = np.mean([feat["log_inc_cases_lag1"], feat["log_inc_cases_lag2"], feat["log_inc_cases_lag3"]])
        feat["inc_cases_roll7"] = np.mean([feat["log_inc_cases_lag1"], feat["log_inc_cases_lag2"], feat["log_inc_cases_lag3"], feat["log_inc_cases_lag7"]])
        feat["inc_fatal_roll3"] = np.mean([feat["log_inc_fatal_lag1"], feat["log_inc_fatal_lag2"], feat["log_inc_fatal_lag3"]])
        feat["inc_fatal_roll7"] = np.mean([feat["log_inc_fatal_lag1"], feat["log_inc_fatal_lag2"], feat["log_inc_fatal_lag3"], feat["log_inc_fatal_lag7"]])

        X_row = pd.DataFrame([feat], columns=feature_cols_b).fillna(0)

        model_cases_b = models_b["log_inc_cases"]
        model_fatal_b = models_b["log_inc_fatal"]

        best_iter_cases = model_cases_b.best_iteration_ if model_cases_b.best_iteration_ is not None else model_cases_b.n_estimators
        best_iter_fatal = model_fatal_b.best_iteration_ if model_fatal_b.best_iteration_ is not None else model_fatal_b.n_estimators

        pred_log_inc_cases = model_cases_b.predict(X_row, num_iteration=best_iter_cases)[0]
        pred_log_inc_fatal = model_fatal_b.predict(X_row, num_iteration=best_iter_fatal)[0]

        pred_inc_cases = max(0.0, np.expm1(pred_log_inc_cases))
        pred_inc_fatal = max(0.0, np.expm1(pred_log_inc_fatal))

        last_cases = hist["ConfirmedCases"].iloc[-1] if len(hist) > 0 else 0.0
        last_fatal = hist["Fatalities"].iloc[-1] if len(hist) > 0 else 0.0

        pred_cases = max(last_cases, last_cases + pred_inc_cases)
        pred_fatal = max(last_fatal, last_fatal + pred_inc_fatal)
        pred_fatal = min(pred_fatal, pred_cases)

        pred_rows.append({
            "ForecastId": int(row["ForecastId"]),
            "geo": geo,
            "Date": current_date,
            "ConfirmedCases": pred_cases,
            "Fatalities": pred_fatal,
            "geo_id": row["geo_id"]
        })

    pred_day_df = pd.DataFrame(pred_rows)
    test_preds_collect.append(pred_day_df[["ForecastId", "ConfirmedCases", "Fatalities"]])
    history_test = pd.concat(
        [history_test, pred_day_df[["geo", "Date", "ConfirmedCases", "Fatalities", "geo_id"]]],
        ignore_index=True,
        sort=False
    )

submission_b = pd.concat(test_preds_collect, ignore_index=True).sort_values("ForecastId").reset_index(drop=True)
submission_b.to_csv("submission_b.csv", index=False)

# -----------------------------
# Ensemble at prediction file level
# -----------------------------
sub_a = pd.read_csv("submission_a.csv")
sub_b = pd.read_csv("submission_b.csv")

blend = sub_a.merge(sub_b, on="ForecastId", suffixes=("_a", "_b"))

wa = 1.0 / max(score_a, 1e-12)
wb = 1.0 / max(score_b, 1e-12)
w_a_default = wa / (wa + wb)

if abs(score_a - score_b) / max(min(score_a, score_b), 1e-12) < 0.03:
    w_a_default = 0.75 * w_a_default + 0.25 * 0.5

grid = [0.2, 0.35, 0.5, 0.65, 0.8]
w_cases = min(grid, key=lambda x: abs(x - w_a_default))
w_fatal = min(grid, key=lambda x: abs(x - (1.0 - w_a_default)))

blend["ConfirmedCases"] = np.expm1(
    w_cases * np.log1p(np.maximum(0, blend["ConfirmedCases_a"].values)) +
    (1.0 - w_cases) * np.log1p(np.maximum(0, blend["ConfirmedCases_b"].values))
)

blend["Fatalities"] = np.expm1(
    w_fatal * np.log1p(np.maximum(0, blend["Fatalities_a"].values)) +
    (1.0 - w_fatal) * np.log1p(np.maximum(0, blend["Fatalities_b"].values))
)

submission = blend[["ForecastId", "ConfirmedCases", "Fatalities"]].copy()
submission = post_process_monotonic(submission, test)
submission.to_csv("submission.csv", index=False)

# -----------------------------
# Validation-time ensemble estimate
# -----------------------------
valid_merge = valid_pred_a[["geo", "Date", "ConfirmedCases", "Fatalities", "pred_ConfirmedCases", "pred_Fatalities"]].merge(
    valid_pred_b.rename(columns={
        "ConfirmedCases": "pred_ConfirmedCases_b",
        "Fatalities": "pred_Fatalities_b"
    }),
    on=["geo", "Date"],
    how="inner"
)

valid_merge["pred_ConfirmedCases_a"] = valid_merge["pred_ConfirmedCases"]
valid_merge["pred_Fatalities_a"] = valid_merge["pred_Fatalities"]

valid_merge["blend_cases"] = np.expm1(
    w_cases * np.log1p(np.maximum(0, valid_merge["pred_ConfirmedCases_a"].values)) +
    (1.0 - w_cases) * np.log1p(np.maximum(0, valid_merge["pred_ConfirmedCases_b"].values))
)
valid_merge["blend_fatal"] = np.expm1(
    w_fatal * np.log1p(np.maximum(0, valid_merge["pred_Fatalities_a"].values)) +
    (1.0 - w_fatal) * np.log1p(np.maximum(0, valid_merge["pred_Fatalities_b"].values))
)

valid_merge = valid_merge.sort_values(["geo", "Date"]).reset_index(drop=True)
valid_merge["blend_cases"] = np.maximum(0, valid_merge["blend_cases"].values)
valid_merge["blend_fatal"] = np.maximum(0, valid_merge["blend_fatal"].values)
valid_merge["blend_cases"] = valid_merge.groupby("geo")["blend_cases"].cummax()
valid_merge["blend_fatal"] = valid_merge.groupby("geo")["blend_fatal"].cummax()
valid_merge["blend_fatal"] = np.minimum(valid_merge["blend_fatal"].values, valid_merge["blend_cases"].values)

final_validation_score = rmsle_multi(
    valid_merge["ConfirmedCases"].values,
    valid_merge["blend_cases"].values,
    valid_merge["Fatalities"].values,
    valid_merge["blend_fatal"].values
)

print(f"Model A Validation Performance: {score_a}")
print(f"Model B Validation Performance: {score_b}")
print(f"Final Validation Performance: {final_validation_score}")
