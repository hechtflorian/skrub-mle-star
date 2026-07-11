
import os
import glob
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA_DIR = "./input"

train_candidates = glob.glob(os.path.join(DATA_DIR, "**", "train.csv"), recursive=True)
train_path = sorted(train_candidates)[0]

train = pd.read_csv(train_path)

train["Province_State"] = train["Province_State"].fillna("None")
train["Country_Region"] = train["Country_Region"].fillna("None")
train["Date"] = pd.to_datetime(train["Date"])
train["geo"] = train["Country_Region"] + "_" + train["Province_State"]

train = train.sort_values(["geo", "Date"]).reset_index(drop=True)

targets = ["ConfirmedCases", "Fatalities"]
lag_list = [1, 2, 3, 7, 14]

def rmsle_multi(y_true_cases, y_pred_cases, y_true_fatal, y_pred_fatal):
    y_true_cases = np.maximum(0, np.asarray(y_true_cases))
    y_pred_cases = np.maximum(0, np.asarray(y_pred_cases))
    y_true_fatal = np.maximum(0, np.asarray(y_true_fatal))
    y_pred_fatal = np.maximum(0, np.asarray(y_pred_fatal))
    err_cases = (np.log1p(y_pred_cases) - np.log1p(y_true_cases)) ** 2
    err_fatal = (np.log1p(y_pred_fatal) - np.log1p(y_true_fatal)) ** 2
    return np.sqrt(np.mean(np.concatenate([err_cases, err_fatal])))

def build_features(df, use_geo=True, use_lags=True):
    df = df.copy()
    df["day"] = (df["Date"] - df["Date"].min()).dt.days
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dow"] = df["Date"].dt.dayofweek

    feature_cols = ["day", "month", "week", "dow"]

    if use_geo:
        df["geo_id"] = df["geo"].astype("category").cat.codes
        feature_cols.append("geo_id")

    if use_lags:
        for t in targets:
            df[f"log_{t}"] = np.log1p(df[t])
            for lag in lag_list:
                df[f"{t}_lag{lag}"] = df.groupby("geo")[f"log_{t}"].shift(lag)
        feature_cols += [c for c in df.columns if "lag" in c]

    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols

def evaluate_ablation(name, use_geo=True, use_lags=True, use_cummax=True):
    df_fe, feature_cols = build_features(train, use_geo=use_geo, use_lags=use_lags)

    max_train_date = df_fe["Date"].max()
    val_days = min(14, max(7, df_fe["Date"].nunique() // 6))
    val_start_date = max_train_date - pd.Timedelta(days=val_days - 1)

    tr_idx = df_fe["Date"] < val_start_date
    va_idx = df_fe["Date"] >= val_start_date

    X_train = df_fe.loc[tr_idx, feature_cols].copy()
    X_valid = df_fe.loc[va_idx, feature_cols].copy()

    valid_pred_df = df_fe.loc[va_idx, ["geo", "Date", "ConfirmedCases", "Fatalities"]].copy()

    for target in targets:
        y_train = np.log1p(df_fe.loc[tr_idx, target].values)

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
        valid_pred_df[f"pred_{target}"] = np.maximum(0, p_valid)

    if use_cummax:
        for target in targets:
            valid_pred_df = valid_pred_df.sort_values(["geo", "Date"]).reset_index(drop=True)
            valid_pred_df[f"pred_{target}"] = valid_pred_df.groupby("geo")[f"pred_{target}"].cummax()

    score = rmsle_multi(
        valid_pred_df["ConfirmedCases"].values,
        valid_pred_df["pred_ConfirmedCases"].values,
        valid_pred_df["Fatalities"].values,
        valid_pred_df["pred_Fatalities"].values
    )
    print(f"{name}: validation RMSLE = {score:.6f}")
    return score

results = {}

results["baseline"] = evaluate_ablation(
    "baseline",
    use_geo=True,
    use_lags=True,
    use_cummax=True
)

results["no_lag_features"] = evaluate_ablation(
    "no_lag_features",
    use_geo=True,
    use_lags=False,
    use_cummax=True
)

results["no_geo_id"] = evaluate_ablation(
    "no_geo_id",
    use_geo=False,
    use_lags=True,
    use_cummax=True
)

results["no_cummax_postprocess"] = evaluate_ablation(
    "no_cummax_postprocess",
    use_geo=True,
    use_lags=True,
    use_cummax=False
)

baseline_score = results["baseline"]
impacts = {}

for k, v in results.items():
    if k == "baseline":
        continue
    impacts[k] = v - baseline_score
    print(f"effect_of_{k}: delta_RMSLE = {impacts[k]:+.6f}")

worst_ablation = max(impacts, key=lambda x: impacts[x])
print(f"most_important_part = {worst_ablation} (largest degradation: {impacts[worst_ablation]:+.6f} RMSLE)")
