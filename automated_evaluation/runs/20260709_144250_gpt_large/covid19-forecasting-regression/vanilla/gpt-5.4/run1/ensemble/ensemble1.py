
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
    all_df[f"log_{t}"] = np.log1p(all_df[t].fillna(0))
    for lag in lag_list:
        all_df[f"{t}_lag{lag}"] = all_df.groupby("geo")[f"log_{t}"].shift(lag)

    all_df[f"{t}_diff_lag1_2"] = all_df[f"{t}_lag1"] - all_df[f"{t}_lag2"]
    all_df[f"{t}_diff_lag1_3"] = all_df[f"{t}_lag1"] - all_df[f"{t}_lag3"]
    all_df[f"{t}_diff_lag1_7"] = all_df[f"{t}_lag1"] - all_df[f"{t}_lag7"]
    all_df[f"{t}_diff_lag7_14"] = all_df[f"{t}_lag7"] - all_df[f"{t}_lag14"]

    all_df[f"{t}_rollmean_3"] = all_df[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3"]].mean(axis=1)
    all_df[f"{t}_rollmean_7"] = all_df[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3", f"{t}_lag7"]].mean(axis=1)
    all_df[f"{t}_rollmax_3"] = all_df[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3"]].max(axis=1)
    all_df[f"{t}_rollmin_3"] = all_df[[f"{t}_lag1", f"{t}_lag2", f"{t}_lag3"]].min(axis=1)

feature_cols = ["day", "month", "week", "dow", "geo_id"] + [
    c for c in all_df.columns if ("lag" in c or "diff_" in c or "roll" in c)
]

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

valid_truth_df = train_fe.loc[va_idx, ["geo", "Date", "ConfirmedCases", "Fatalities"]].copy()

def rmsle_multi(y_true_cases, y_pred_cases, y_true_fatal, y_pred_fatal):
    y_true_cases = np.maximum(0, np.asarray(y_true_cases))
    y_pred_cases = np.maximum(0, np.asarray(y_pred_cases))
    y_true_fatal = np.maximum(0, np.asarray(y_true_fatal))
    y_pred_fatal = np.maximum(0, np.asarray(y_pred_fatal))
    err_cases = (np.log1p(y_pred_cases) - np.log1p(y_true_cases)) ** 2
    err_fatal = (np.log1p(y_pred_fatal) - np.log1p(y_true_fatal)) ** 2
    return np.sqrt(np.mean(np.concatenate([err_cases, err_fatal])))

def build_model_predictions(model_name, n_estimators, learning_rate, num_leaves, subsample, colsample_bytree, extra_seed):
    valid_pred_df = valid_truth_df.copy()
    test_pred_df = test_fe[["ForecastId", "geo", "Date"]].copy()

    for target in targets:
        y_train = np.log1p(train_fe.loc[tr_idx, target].values)
        y_valid = np.log1p(train_fe.loc[va_idx, target].values)

        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42 + extra_seed
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        best_iter = model.best_iteration_ if model.best_iteration_ is not None else model.n_estimators
        p_valid = np.expm1(model.predict(X_valid, num_iteration=best_iter))
        p_test = np.expm1(model.predict(test_fe[feature_cols], num_iteration=best_iter))

        valid_pred_df[f"pred_{target}"] = np.maximum(0, p_valid)
        test_pred_df[target] = np.maximum(0, p_test)

    model_val_score = rmsle_multi(
        valid_pred_df["ConfirmedCases"].values,
        valid_pred_df["pred_ConfirmedCases"].values,
        valid_pred_df["Fatalities"].values,
        valid_pred_df["pred_Fatalities"].values
    )

    submission_df = pd.DataFrame({
        "ForecastId": test_pred_df["ForecastId"].astype(int),
        "ConfirmedCases": np.maximum(0, test_pred_df["ConfirmedCases"].values),
        "Fatalities": np.maximum(0, test_pred_df["Fatalities"].values)
    }).sort_values("ForecastId").reset_index(drop=True)

    submission_df.to_csv(f"submission_{model_name}.csv", index=False)
    valid_pred_df[["geo", "Date", "pred_ConfirmedCases", "pred_Fatalities"]].to_csv(
        f"validation_preds_{model_name}.csv", index=False
    )

    return valid_pred_df, submission_df, model_val_score

valid_a, submission_a, score_a = build_model_predictions(
    model_name="a",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    extra_seed=0
)

valid_b, submission_b, score_b = build_model_predictions(
    model_name="b",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.85,
    colsample_bytree=0.85,
    extra_seed=17
)

last_obs = (
    train_fe.loc[tr_idx, ["geo", "ConfirmedCases", "Fatalities"]]
    .sort_index()
    .groupby("geo")
    .last()
    .reset_index()
    .rename(columns={
        "ConfirmedCases": "last_train_cases",
        "Fatalities": "last_train_fatal"
    })
)

def make_horizon_bucket(h):
    if h <= 7:
        return "short"
    elif h <= 14:
        return "mid"
    else:
        return "long"

def assign_severity_buckets(last_train_cases_series):
    logv = np.log1p(last_train_cases_series.fillna(0).values.astype(float))
    uniq = np.unique(logv)
    if len(uniq) < 3:
        bins = [-np.inf, np.quantile(logv, 0.5), np.inf]
        labels = ["cold", "mid"]
        out = pd.cut(logv, bins=bins, labels=labels, include_lowest=True, duplicates="drop")
        out = pd.Series(out.astype(str)).replace("nan", "mid")
        out = out.map(lambda x: "cold" if x == "cold" else "hot")
        return out.values
    q1, q2 = np.quantile(logv, [1/3, 2/3])
    if q1 == q2:
        q1, q2 = np.quantile(logv, [0.25, 0.75])
    if q1 == q2:
        return np.where(logv <= q1, "cold", "hot")
    sev = np.where(logv <= q1, "cold", np.where(logv <= q2, "mid", "hot"))
    return sev

valid_merge = valid_truth_df.merge(
    valid_a[["geo", "Date", "pred_ConfirmedCases", "pred_Fatalities"]].rename(columns={
        "pred_ConfirmedCases": "pred_cases_a",
        "pred_Fatalities": "pred_fatal_a"
    }),
    on=["geo", "Date"],
    how="left"
).merge(
    valid_b[["geo", "Date", "pred_ConfirmedCases", "pred_Fatalities"]].rename(columns={
        "pred_ConfirmedCases": "pred_cases_b",
        "pred_Fatalities": "pred_fatal_b"
    }),
    on=["geo", "Date"],
    how="left"
).merge(
    last_obs,
    on="geo",
    how="left"
)

valid_merge["horizon"] = valid_merge.groupby("geo")["Date"].rank(method="dense").astype(int)
valid_merge["horizon_bucket"] = valid_merge["horizon"].apply(make_horizon_bucket)
valid_merge["severity_bucket"] = assign_severity_buckets(valid_merge["last_train_cases"])
valid_merge["cell"] = valid_merge["horizon_bucket"] + "|" + valid_merge["severity_bucket"]

test_meta = test[["ForecastId", "geo", "Date"]].copy().merge(last_obs, on="geo", how="left")
test_meta = test_meta.sort_values(["geo", "Date"]).reset_index(drop=True)
test_meta["horizon"] = test_meta.groupby("geo")["Date"].rank(method="dense").astype(int)
test_meta["horizon_bucket"] = test_meta["horizon"].apply(make_horizon_bucket)
test_meta["severity_bucket"] = assign_severity_buckets(test_meta["last_train_cases"])
test_meta["cell"] = test_meta["horizon_bucket"] + "|" + test_meta["severity_bucket"]

weights_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

def blended_from_weight(pred_a, pred_b, w):
    pa = np.maximum(0, np.asarray(pred_a, dtype=float))
    pb = np.maximum(0, np.asarray(pred_b, dtype=float))
    return np.expm1(w * np.log1p(pa) + (1.0 - w) * np.log1p(pb))

def rmsle_single(y_true, y_pred):
    yt = np.maximum(0, np.asarray(y_true, dtype=float))
    yp = np.maximum(0, np.asarray(y_pred, dtype=float))
    return np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2))

def best_weight_for_subset(df, true_col, pred_a_col, pred_b_col):
    best_w = 0.5
    best_score = np.inf
    for w in weights_grid:
        pred = blended_from_weight(df[pred_a_col].values, df[pred_b_col].values, w)
        score = rmsle_single(df[true_col].values, pred)
        if score < best_score:
            best_score = score
            best_w = w
    return best_w, best_score

global_w_cases, _ = best_weight_for_subset(valid_merge, "ConfirmedCases", "pred_cases_a", "pred_cases_b")
global_w_fatal, _ = best_weight_for_subset(valid_merge, "Fatalities", "pred_fatal_a", "pred_fatal_b")

min_cell_size = 20
w_cases_lookup = {}
w_fatal_lookup = {}

horizon_levels = ["short", "mid", "long"]
severity_levels = ["cold", "mid", "hot"]

for hb in horizon_levels:
    for sb in severity_levels:
        cell_df = valid_merge[(valid_merge["horizon_bucket"] == hb) & (valid_merge["severity_bucket"] == sb)]
        if len(cell_df) >= min_cell_size:
            w_cases_lookup[(hb, sb)] = best_weight_for_subset(cell_df, "ConfirmedCases", "pred_cases_a", "pred_cases_b")[0]
            w_fatal_lookup[(hb, sb)] = best_weight_for_subset(cell_df, "Fatalities", "pred_fatal_a", "pred_fatal_b")[0]
        else:
            sev_df = valid_merge[valid_merge["severity_bucket"] == sb]
            hor_df = valid_merge[valid_merge["horizon_bucket"] == hb]

            if len(sev_df) >= min_cell_size:
                w_cases_lookup[(hb, sb)] = best_weight_for_subset(sev_df, "ConfirmedCases", "pred_cases_a", "pred_cases_b")[0]
                w_fatal_lookup[(hb, sb)] = best_weight_for_subset(sev_df, "Fatalities", "pred_fatal_a", "pred_fatal_b")[0]
            elif len(hor_df) >= min_cell_size:
                w_cases_lookup[(hb, sb)] = best_weight_for_subset(hor_df, "ConfirmedCases", "pred_cases_a", "pred_cases_b")[0]
                w_fatal_lookup[(hb, sb)] = best_weight_for_subset(hor_df, "Fatalities", "pred_fatal_a", "pred_fatal_b")[0]
            else:
                w_cases_lookup[(hb, sb)] = global_w_cases
                w_fatal_lookup[(hb, sb)] = global_w_fatal

valid_merge["w_cases"] = valid_merge.apply(lambda r: w_cases_lookup.get((r["horizon_bucket"], r["severity_bucket"]), global_w_cases), axis=1)
valid_merge["w_fatal"] = valid_merge.apply(lambda r: w_fatal_lookup.get((r["horizon_bucket"], r["severity_bucket"]), global_w_fatal), axis=1)

valid_merge["blend_cases"] = np.expm1(
    valid_merge["w_cases"].values * np.log1p(np.maximum(0, valid_merge["pred_cases_a"].values)) +
    (1.0 - valid_merge["w_cases"].values) * np.log1p(np.maximum(0, valid_merge["pred_cases_b"].values))
)
valid_merge["blend_fatal"] = np.expm1(
    valid_merge["w_fatal"].values * np.log1p(np.maximum(0, valid_merge["pred_fatal_a"].values)) +
    (1.0 - valid_merge["w_fatal"].values) * np.log1p(np.maximum(0, valid_merge["pred_fatal_b"].values))
)

valid_merge["blend_cases"] = np.maximum(0, valid_merge["blend_cases"])
valid_merge["blend_fatal"] = np.maximum(0, valid_merge["blend_fatal"])

valid_merge = valid_merge.sort_values(["geo", "Date"]).reset_index(drop=True)
valid_merge["blend_cases"] = valid_merge.groupby("geo")["blend_cases"].cummax()
valid_merge["blend_fatal"] = valid_merge.groupby("geo")["blend_fatal"].cummax()
valid_merge["blend_fatal"] = np.minimum(valid_merge["blend_fatal"], valid_merge["blend_cases"])

final_validation_score = rmsle_multi(
    valid_merge["ConfirmedCases"].values,
    valid_merge["blend_cases"].values,
    valid_merge["Fatalities"].values,
    valid_merge["blend_fatal"].values
)

submission_a = pd.read_csv("submission_a.csv")
submission_b = pd.read_csv("submission_b.csv")

test_blend = test_meta.merge(
    submission_a.rename(columns={
        "ConfirmedCases": "ConfirmedCases_a",
        "Fatalities": "Fatalities_a"
    }),
    on="ForecastId",
    how="left"
).merge(
    submission_b.rename(columns={
        "ConfirmedCases": "ConfirmedCases_b",
        "Fatalities": "Fatalities_b"
    }),
    on="ForecastId",
    how="left"
)

test_blend["w_cases"] = test_blend.apply(lambda r: w_cases_lookup.get((r["horizon_bucket"], r["severity_bucket"]), global_w_cases), axis=1)
test_blend["w_fatal"] = test_blend.apply(lambda r: w_fatal_lookup.get((r["horizon_bucket"], r["severity_bucket"]), global_w_fatal), axis=1)

test_blend["ConfirmedCases"] = np.expm1(
    test_blend["w_cases"].values * np.log1p(np.maximum(0, test_blend["ConfirmedCases_a"].values)) +
    (1.0 - test_blend["w_cases"].values) * np.log1p(np.maximum(0, test_blend["ConfirmedCases_b"].values))
)
test_blend["Fatalities"] = np.expm1(
    test_blend["w_fatal"].values * np.log1p(np.maximum(0, test_blend["Fatalities_a"].values)) +
    (1.0 - test_blend["w_fatal"].values) * np.log1p(np.maximum(0, test_blend["Fatalities_b"].values))
)

test_blend["ConfirmedCases"] = np.maximum(0, test_blend["ConfirmedCases"])
test_blend["Fatalities"] = np.maximum(0, test_blend["Fatalities"])

test_blend = test_blend.sort_values(["geo", "Date"]).reset_index(drop=True)
test_blend["ConfirmedCases"] = test_blend.groupby("geo")["ConfirmedCases"].cummax()
test_blend["Fatalities"] = test_blend.groupby("geo")["Fatalities"].cummax()
test_blend["Fatalities"] = np.minimum(test_blend["Fatalities"], test_blend["ConfirmedCases"])

submission = pd.DataFrame({
    "ForecastId": test_blend["ForecastId"].astype(int),
    "ConfirmedCases": test_blend["ConfirmedCases"].values,
    "Fatalities": test_blend["Fatalities"].values
})

submission["ConfirmedCases"] = np.maximum(0, submission["ConfirmedCases"])
submission["Fatalities"] = np.maximum(0, submission["Fatalities"])

submission = submission.sort_values("ForecastId").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
