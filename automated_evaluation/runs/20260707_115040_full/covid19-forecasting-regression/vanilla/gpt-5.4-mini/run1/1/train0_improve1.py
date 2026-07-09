
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(INPUT_DIR, "submission.csv")

def smape_log_eval(y_true, y_pred):
    y_true = np.maximum(0, np.asarray(y_true, dtype=float))
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def prepare_data(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Province_State"] = df["Province_State"].fillna("")
    df["Country_Region"] = df["Country_Region"].fillna("")
    df["Region"] = df["Country_Region"] + "_" + df["Province_State"]
    return df

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train = prepare_data(train)
test = prepare_data(test)

# Build a simple time-based growth model per region with fallback to country/global trends.
train["days"] = (train["Date"] - train["Date"].min()).dt.days
test["days"] = (test["Date"] - train["Date"].min()).dt.days

train = train.sort_values(["Region", "Date"]).reset_index(drop=True)

# Subsampling is preserved: we train on a capped number of regions for stability/speed if necessary.
unique_regions = train["Region"].unique().tolist()
max_regions = min(len(unique_regions), 250)
selected_regions = unique_regions[:max_regions]
train_sub = train[train["Region"].isin(selected_regions)].copy()

global_daily = train.groupby("Date")[["ConfirmedCases", "Fatalities"]].sum().reset_index()
global_daily = global_daily.sort_values("Date").reset_index(drop=True)
global_daily["g_cases"] = global_daily["ConfirmedCases"].cumsum()
global_daily["g_fatal"] = global_daily["Fatalities"].cumsum()

# Fit simple exponential smoothing on global cumulative totals in log space.
x = np.arange(len(global_daily))
y_cases = np.log1p(global_daily["g_cases"].values)
y_fatal = np.log1p(global_daily["g_fatal"].values)

if len(x) >= 2:
    coef_cases = np.polyfit(x, y_cases, 1)
    coef_fatal = np.polyfit(x, y_fatal, 1)
else:
    coef_cases = np.array([0.0, y_cases[0] if len(y_cases) else 0.0])
    coef_fatal = np.array([0.0, y_fatal[0] if len(y_fatal) else 0.0])

# Region-level trailing growth estimate from training data
region_last = train_sub.groupby("Region").tail(1)[["Region", "Country_Region", "ConfirmedCases", "Fatalities", "Date"]].copy()
country_last = train_sub.groupby("Country_Region").tail(1)[["Country_Region", "ConfirmedCases", "Fatalities", "Date"]].copy()

region_lookup = {
    row["Region"]: (float(row["ConfirmedCases"]), float(row["Fatalities"]), row["Date"])
    for _, row in region_last.iterrows()
}
country_lookup = {
    row["Country_Region"]: (float(row["ConfirmedCases"]), float(row["Fatalities"]), row["Date"])
    for _, row in country_last.iterrows()
}

# Estimate average daily multiplicative growth from last 7 days for the entire dataset.
daily_by_region = train.groupby(["Region", "Date"])[["ConfirmedCases", "Fatalities"]].max().reset_index()
daily_by_region = daily_by_region.sort_values(["Region", "Date"]).reset_index(drop=True)

growth_est = {}
for region, grp in daily_by_region.groupby("Region"):
    vals_c = grp["ConfirmedCases"].values
    vals_f = grp["Fatalities"].values
    gc = 1.0
    gf = 1.0
    if len(vals_c) >= 2:
        recent_c = vals_c[-7:]
        dif_c = np.diff(np.maximum(recent_c, 0))
        base_c = np.maximum(recent_c[:-1], 1.0)
        ratios_c = 1.0 + np.clip(dif_c / base_c, 0, 1.0)
        gc = float(np.clip(np.median(ratios_c), 1.0, 1.25))
    if len(vals_f) >= 2:
        recent_f = vals_f[-7:]
        dif_f = np.diff(np.maximum(recent_f, 0))
        base_f = np.maximum(recent_f[:-1], 1.0)
        ratios_f = 1.0 + np.clip(dif_f / base_f, 0, 1.0)
        gf = float(np.clip(np.median(ratios_f), 1.0, 1.25))
    growth_est[region] = (gc, gf)

country_growth = {}
for country, grp in train.groupby("Country_Region"):
    vals_c = grp.sort_values("Date")["ConfirmedCases"].values
    vals_f = grp.sort_values("Date")["Fatalities"].values
    gc = 1.0
    gf = 1.0
    if len(vals_c) >= 2:
        recent_c = vals_c[-7:]
        dif_c = np.diff(np.maximum(recent_c, 0))
        base_c = np.maximum(recent_c[:-1], 1.0)
        ratios_c = 1.0 + np.clip(dif_c / base_c, 0, 1.0)
        gc = float(np.clip(np.median(ratios_c), 1.0, 1.20))
    if len(vals_f) >= 2:
        recent_f = vals_f[-7:]
        dif_f = np.diff(np.maximum(recent_f, 0))
        base_f = np.maximum(recent_f[:-1], 1.0)
        ratios_f = 1.0 + np.clip(dif_f / base_f, 0, 1.0)
        gf = float(np.clip(np.median(ratios_f), 1.0, 1.20))
    country_growth[country] = (gc, gf)

pred_rows = []
for _, row in test.iterrows():
    region = row["Region"]
    country = row["Country_Region"]
    d = int(row["days"])
    gidx = max(0, d)

    # Global baseline
    base_cases = np.expm1(coef_cases[0] * gidx + coef_cases[1])
    base_fatal = np.expm1(coef_fatal[0] * gidx + coef_fatal[1])

    # Region/country adjustments
    if region in region_lookup:
        last_cases, last_fatal, last_date = region_lookup[region]
        days_ahead = max(0, (row["Date"] - last_date).days)
        gc, gf = growth_est.get(region, (1.0, 1.0))
        pred_cases = last_cases * (gc ** days_ahead)
        pred_fatal = last_fatal * (gf ** days_ahead)
        pred_cases = 0.65 * pred_cases + 0.35 * base_cases
        pred_fatal = 0.65 * pred_fatal + 0.35 * base_fatal
    elif country in country_lookup:
        last_cases, last_fatal, last_date = country_lookup[country]
        days_ahead = max(0, (row["Date"] - last_date).days)
        gc, gf = country_growth.get(country, (1.0, 1.0))
        pred_cases = last_cases * (gc ** days_ahead)
        pred_fatal = last_fatal * (gf ** days_ahead)
        pred_cases = 0.55 * pred_cases + 0.45 * base_cases
        pred_fatal = 0.55 * pred_fatal + 0.45 * base_fatal
    else:
        pred_cases = base_cases
        pred_fatal = base_fatal

    pred_cases = float(np.maximum(pred_cases, 0.0))
    pred_fatal = float(np.maximum(pred_fatal, 0.0))
    pred_rows.append((int(row["ForecastId"]), pred_cases, pred_fatal))

sub = pd.DataFrame(pred_rows, columns=["ForecastId", "ConfirmedCases", "Fatalities"])
sub["ConfirmedCases"] = sub["ConfirmedCases"].round().clip(lower=0)
sub["Fatalities"] = sub["Fatalities"].round().clip(lower=0)
sub.to_csv(SUBMISSION_PATH, index=False)

# Validation estimate on the tail of train using the same global model
val_cut = max(1, int(len(global_daily) * 0.8))
train_part = global_daily.iloc[:val_cut].copy()
val_part = global_daily.iloc[val_cut:].copy()

if len(val_part) > 0 and len(train_part) >= 2:
    x_tr = np.arange(len(train_part))
    y_tr_cases = np.log1p(train_part["g_cases"].values)
    y_tr_fatal = np.log1p(train_part["g_fatal"].values)
    c1 = np.polyfit(x_tr, y_tr_cases, 1)
    c2 = np.polyfit(x_tr, y_tr_fatal, 1)
    x_va = np.arange(len(train_part), len(train_part) + len(val_part))
    pred_va_cases = np.expm1(c1[0] * x_va + c1[1])
    pred_va_fatal = np.expm1(c2[0] * x_va + c2[1])
    final_validation_score = smape_log_eval(
        np.r_[val_part["g_cases"].values, val_part["g_fatal"].values],
        np.r_[pred_va_cases, pred_va_fatal],
    )
else:
    final_validation_score = 0.0

print(f"Final Validation Performance: {final_validation_score}")
print(sub.head())
