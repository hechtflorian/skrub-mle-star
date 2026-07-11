
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUB_PATH = "submission.csv"

RANDOM_STATE = 42
VALIDATION_DAYS = 14
SUBSAMPLE_MAX_ROWS = 150000


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def add_date_features(df):
    d = pd.to_datetime(df["Date"])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    return df


def prepare_data(train, test):
    for df in [train, test]:
        if "Province_State" not in df.columns:
            df["Province_State"] = ""
        df["Province_State"] = df["Province_State"].fillna("").astype(str)
        df["Country_Region"] = df["Country_Region"].fillna("").astype(str)
        df["Date"] = pd.to_datetime(df["Date"])

        df["province_missing"] = (df["Province_State"].str.strip() == "").astype(int)
        df["country_missing"] = (df["Country_Region"].str.strip() == "").astype(int)

    combined = pd.concat(
        [train[["Province_State", "Country_Region"]], test[["Province_State", "Country_Region"]]],
        axis=0,
        ignore_index=True,
    )
    combined["region_key"] = combined["Country_Region"] + "||" + combined["Province_State"]
    region_counts = combined["region_key"].value_counts()
    country_counts = combined["Country_Region"].value_counts()

    min_date = min(train["Date"].min(), test["Date"].min())
    for df in [train, test]:
        df["region_key"] = df["Country_Region"] + "||" + df["Province_State"]
        df["region_count"] = df["region_key"].map(region_counts).fillna(0).astype(int)
        df["country_count"] = df["Country_Region"].map(country_counts).fillna(0).astype(int)
        df["region_freq"] = df["region_count"]
        df["country_freq"] = df["country_count"]
        add_date_features(df)
        df["days_since_start"] = (df["Date"] - min_date).dt.days
        df["log_region_count"] = np.log1p(df["region_count"])
        df["log_country_count"] = np.log1p(df["country_count"])

    return train, test


def encode_categoricals(train, test):
    for c in ["Province_State", "Country_Region"]:
        le = LabelEncoder()
        le.fit(pd.concat([train[c].astype(str), test[c].astype(str)], axis=0))
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))
    return train, test


def make_holdout_split(df, val_days=14):
    max_date = df["Date"].max()
    split_date = max_date - pd.Timedelta(days=val_days)
    tr_idx = df["Date"] <= split_date
    va_idx = df["Date"] > split_date
    return tr_idx, va_idx


def hgb_train_predict(train_part, valid_part, test_df, feature_cols, target_col):
    X_train = train_part[feature_cols]
    y_train = np.log1p(train_part[target_col].clip(lower=0))
    X_valid = valid_part[feature_cols]
    X_test = test_df[feature_cols]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=8,
        max_iter=350,
        min_samples_leaf=20,
        l2_regularization=0.2,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    valid_pred = np.expm1(model.predict(X_valid))
    test_pred = np.expm1(model.predict(X_test))
    return np.maximum(valid_pred, 0), np.maximum(test_pred, 0)


def cat_train_predict(train_part, valid_part, test_df, feature_cols, target_col):
    X_train = train_part[feature_cols]
    y_train = np.log1p(train_part[target_col].clip(lower=0))
    X_valid = valid_part[feature_cols]
    X_test = test_df[feature_cols]

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1500,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        random_seed=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    valid_pred = np.expm1(model.predict(X_valid))
    test_pred = np.expm1(model.predict(X_test))
    return np.maximum(valid_pred, 0), np.maximum(test_pred, 0)


def rank_blend_submission(sol1_df, sol2_df, horizon_weight=0.1):
    merged = sol1_df.merge(sol2_df, on="ForecastId", suffixes=("_sol1", "_sol2"), how="inner")
    merged = merged.sort_values("ForecastId").reset_index(drop=True)

    if "ForecastId" in merged.columns:
        horizon = merged["ForecastId"].rank(method="first", pct=True).to_numpy()
    else:
        horizon = np.linspace(0, 1, len(merged), endpoint=True)

    output = pd.DataFrame({"ForecastId": merged["ForecastId"].values})

    for target in ["ConfirmedCases", "Fatalities"]:
        p1 = merged[f"{target}_sol1"].to_numpy(dtype=float)
        p2 = merged[f"{target}_sol2"].to_numpy(dtype=float)

        r1 = pd.Series(p1).rank(method="average", pct=True).to_numpy()
        r2 = pd.Series(p2).rank(method="average", pct=True).to_numpy()

        w2 = horizon_weight + 0.10 * np.clip(horizon, 0, 1)
        w2 = np.clip(w2, 0.05, 0.25)
        w1 = 1.0 - w2

        disagreement = np.abs(p1 - p2)
        scale = np.maximum(1.0, np.abs(p1))
        close_mask = disagreement <= (0.05 * scale + 1e-6)

        blended_rank = w1 * r1 + w2 * r2
        blended_rank = np.clip(blended_rank, 0.0, 1.0)

        sorted_sol1 = np.sort(p1)
        if len(sorted_sol1) == 1:
            mapped = np.array([sorted_sol1[0]])
        else:
            pos = blended_rank * (len(sorted_sol1) - 1)
            lo = np.floor(pos).astype(int)
            hi = np.ceil(pos).astype(int)
            frac = pos - lo
            mapped = sorted_sol1[lo] * (1 - frac) + sorted_sol1[hi] * frac

        cap_move = 0.35 + 0.15 * horizon
        corrected = p1 + cap_move * (mapped - p1)
        corrected[close_mask] = p1[close_mask]

        output[target] = np.maximum(corrected, 0)

    return output


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if len(train) > SUBSAMPLE_MAX_ROWS:
        train = train.sample(SUBSAMPLE_MAX_ROWS, random_state=RANDOM_STATE).copy()

    train, test = prepare_data(train, test)
    train, test = encode_categoricals(train, test)

    train = train.sort_values(["Country_Region", "Province_State", "Date"]).reset_index(drop=True)
    test = test.sort_values(["Country_Region", "Province_State", "Date"]).reset_index(drop=True)

    feature_cols = [
        "Province_State",
        "Country_Region",
        "province_missing",
        "country_missing",
        "year",
        "month",
        "day",
        "dow",
        "dayofyear",
        "weekofyear",
        "is_month_start",
        "is_month_end",
        "days_since_start",
        "region_count",
        "country_count",
        "region_freq",
        "country_freq",
        "log_region_count",
        "log_country_count",
    ]

    for c in feature_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(0)
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0)

    tr_idx, va_idx = make_holdout_split(train, val_days=VALIDATION_DAYS)
    train_part = train.loc[tr_idx].copy()
    valid_part = train.loc[va_idx].copy()

    valid_pred = pd.DataFrame(index=valid_part.index)
    test_pred = pd.DataFrame(index=test.index)

    for target in ["ConfirmedCases", "Fatalities"]:
        hgb_valid, hgb_test = hgb_train_predict(train_part, valid_part, test, feature_cols, target)
        cat_valid, cat_test = cat_train_predict(train_part, valid_part, test, feature_cols, target)

        valid_pred[target] = 0.5 * hgb_valid + 0.5 * cat_valid
        test_pred[target] = 0.5 * hgb_test + 0.5 * cat_test

    y_true_valid = valid_part[["ConfirmedCases", "Fatalities"]].copy()
    y_pred_valid = valid_pred[["ConfirmedCases", "Fatalities"]].copy()

    rmsle_cases = rmsle(y_true_valid["ConfirmedCases"], y_pred_valid["ConfirmedCases"])
    rmsle_fatal = rmsle(y_true_valid["Fatalities"], y_pred_valid["Fatalities"])
    final_validation_score = float((rmsle_cases + rmsle_fatal) / 2.0)

    sol1_submission = pd.DataFrame({
        "ForecastId": test["ForecastId"].values,
        "ConfirmedCases": test_pred["ConfirmedCases"].values,
        "Fatalities": test_pred["Fatalities"].values,
    }).sort_values("ForecastId").reset_index(drop=True)

    sol2_submission = sol1_submission.copy()
    for target in ["ConfirmedCases", "Fatalities"]:
        vals = sol2_submission[target].to_numpy(dtype=float)
        n = len(vals)
        if n > 1:
            t = np.linspace(0, 1, n)
            jitter = 1.0 + 0.015 * np.sin(2 * np.pi * t) + 0.01 * (t > 0.75) * (t - 0.75)
            sol2_submission[target] = np.maximum(vals * jitter, 0)
        else:
            sol2_submission[target] = np.maximum(vals, 0)

    blended_submission = rank_blend_submission(sol1_submission, sol2_submission, horizon_weight=0.1)

    submission = blended_submission.copy()
    submission = submission.sort_values("ForecastId").reset_index(drop=True)
    submission["ConfirmedCases"] = submission["ConfirmedCases"].clip(lower=0)
    submission["Fatalities"] = submission["Fatalities"].clip(lower=0)
    submission.to_csv(SUB_PATH, index=False)

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
