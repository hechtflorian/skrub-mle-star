
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUB_PATH = "submission.csv"

RANDOM_STATE = 42
SUBSAMPLE_MAX_ROWS = 150000  # keep subsampling if exists
VALIDATION_DAYS = 14


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def safe_col(df, col, default=np.nan):
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def add_date_features(df):
    d = pd.to_datetime(df["Date"])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    return df


def prepare_data(train_df, test_df):
    for df in [train_df, test_df]:
        df["Province_State"] = safe_col(df, "Province_State", "")
        df["Province_State"] = df["Province_State"].fillna("").astype(str)
        df["Country_Region"] = safe_col(df, "Country_Region", "").fillna("").astype(str)
        df["Date"] = pd.to_datetime(df["Date"])
        df["province_missing"] = (df["Province_State"].str.strip() == "").astype(int)
        df["country_missing"] = (df["Country_Region"].str.strip() == "").astype(int)

    combined = pd.concat(
        [
            train_df[["Province_State", "Country_Region"]],
            test_df[["Province_State", "Country_Region"]],
        ],
        axis=0,
        ignore_index=True,
    )
    combined["region_key"] = combined["Country_Region"] + "||" + combined["Province_State"]
    region_counts = combined["region_key"].value_counts()
    country_counts = combined["Country_Region"].value_counts()

    for df in [train_df, test_df]:
        df["region_key"] = df["Country_Region"] + "||" + df["Province_State"]
        df["region_count"] = df["region_key"].map(region_counts).fillna(0).astype(int)
        df["country_count"] = df["Country_Region"].map(country_counts).fillna(0).astype(int)
        df = add_date_features(df)

    # label-like frequency encoding
    train_df["region_freq"] = train_df["region_key"].map(region_counts).fillna(0).astype(int)
    test_df["region_freq"] = test_df["region_key"].map(region_counts).fillna(0).astype(int)
    train_df["country_freq"] = train_df["Country_Region"].map(country_counts).fillna(0).astype(int)
    test_df["country_freq"] = test_df["Country_Region"].map(country_counts).fillna(0).astype(int)

    return train_df, test_df


def make_features(df):
    df = df.copy()
    df = add_date_features(df)

    # robust categorical encoding
    df["Province_State"] = df["Province_State"].fillna("").astype(str)
    df["Country_Region"] = df["Country_Region"].fillna("").astype(str)
    df["region_key"] = df["Country_Region"] + "||" + df["Province_State"]

    # simple time trend features
    min_date = df["Date"].min()
    df["days_since_start"] = (df["Date"] - min_date).dt.days

    # interaction / hierarchy
    df["log_region_count"] = np.log1p(df["region_count"])
    df["log_country_count"] = np.log1p(df["country_count"])
    return df


def build_country_level_totals(train_df, y_col):
    # Aggregate across province/state to country level, since test includes provinces as separate rows.
    cols = ["Country_Region", "Date", y_col]
    tmp = train_df[cols].copy()
    tmp = tmp.groupby(["Country_Region", "Date"], as_index=False)[y_col].sum()
    return tmp


def train_and_validate(train_df, feature_cols, target_col):
    train_df = train_df.sort_values("Date").reset_index(drop=True)

    unique_dates = np.array(sorted(train_df["Date"].unique()))
    if len(unique_dates) <= VALIDATION_DAYS + 1:
        split_date = unique_dates[max(1, len(unique_dates) // 2)]
    else:
        split_date = unique_dates[-VALIDATION_DAYS]

    tr_idx = train_df["Date"] < split_date
    va_idx = train_df["Date"] >= split_date

    X_train = train_df.loc[tr_idx, feature_cols]
    y_train = np.log1p(train_df.loc[tr_idx, target_col].clip(lower=0))

    X_valid = train_df.loc[va_idx, feature_cols]
    y_valid = train_df.loc[va_idx, target_col].clip(lower=0).values

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

    pred_valid = np.expm1(model.predict(X_valid))
    pred_valid = np.maximum(pred_valid, 0)

    score = rmsle(y_valid, pred_valid)
    return model, score


def fit_full_and_predict(train_df, test_df, feature_cols, target_col):
    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df[target_col].clip(lower=0))

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
    pred_test = np.expm1(model.predict(X_test))
    pred_test = np.maximum(pred_test, 0)
    return pred_test


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # handle missing columns robustly
    if "Province_State" not in train.columns:
        train["Province_State"] = ""
    if "Province_State" not in test.columns:
        test["Province_State"] = ""

    train["Province_State"] = train["Province_State"].fillna("").astype(str)
    test["Province_State"] = test["Province_State"].fillna("").astype(str)
    train["Country_Region"] = train["Country_Region"].fillna("").astype(str)
    test["Country_Region"] = test["Country_Region"].fillna("").astype(str)

    # subsampling if exists
    if len(train) > SUBSAMPLE_MAX_ROWS:
        train = train.sample(SUBSAMPLE_MAX_ROWS, random_state=RANDOM_STATE).copy()

    # sort before time split
    train = train.sort_values(["Country_Region", "Province_State", "Date"]).reset_index(drop=True)
    test = test.sort_values(["Country_Region", "Province_State", "Date"]).reset_index(drop=True)

    # feature engineering
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    for df in [train, test]:
        df["province_missing"] = (df["Province_State"].fillna("").astype(str).str.strip() == "").astype(int)
        df["country_missing"] = (df["Country_Region"].fillna("").astype(str).str.strip() == "").astype(int)

    # counts for encoding
    combined_regions = pd.concat(
        [
            train[["Country_Region", "Province_State"]],
            test[["Country_Region", "Province_State"]],
        ],
        ignore_index=True,
    )
    combined_regions["region_key"] = combined_regions["Country_Region"].astype(str) + "||" + combined_regions["Province_State"].astype(str)
    region_counts = combined_regions["region_key"].value_counts()
    country_counts = combined_regions["Country_Region"].value_counts()

    for df in [train, test]:
        df["region_key"] = df["Country_Region"].astype(str) + "||" + df["Province_State"].astype(str)
        df["region_count"] = df["region_key"].map(region_counts).fillna(0).astype(int)
        df["country_count"] = df["Country_Region"].map(country_counts).fillna(0).astype(int)
        df = add_date_features(df)

    # advanced time features
    min_date = min(train["Date"].min(), test["Date"].min())
    for df in [train, test]:
        df["days_since_start"] = (df["Date"] - min_date).dt.days
        df["log_region_count"] = np.log1p(df["region_count"])
        df["log_country_count"] = np.log1p(df["country_count"])

    # country-level features improve over naive per-row date-only model
    # We'll train row-level models using these features and target as cumulative values
    feature_cols = [
        "province_missing",
        "country_missing",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "is_month_start",
        "is_month_end",
        "days_since_start",
        "region_count",
        "country_count",
        "log_region_count",
        "log_country_count",
    ]

    # ensure date features exist in both
    train = add_date_features(train)
    test = add_date_features(test)
    train["days_since_start"] = (train["Date"] - min_date).dt.days
    test["days_since_start"] = (test["Date"] - min_date).dt.days
    train["log_region_count"] = np.log1p(train["region_count"])
    test["log_region_count"] = np.log1p(test["region_count"])
    train["log_country_count"] = np.log1p(train["country_count"])
    test["log_country_count"] = np.log1p(test["country_count"])

    # fill any remaining missing feature values
    for df in [train, test]:
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # validation for ConfirmedCases
    conf_model, conf_score = train_and_validate(train, feature_cols, "ConfirmedCases")
    fat_model, fat_score = train_and_validate(train, feature_cols, "Fatalities")
    final_validation_score = (conf_score + fat_score) / 2.0
    print(f"Final Validation Performance: {final_validation_score}")

    # fit full models and predict
    conf_pred = fit_full_and_predict(train, test, feature_cols, "ConfirmedCases")
    fat_pred = fit_full_and_predict(train, test, feature_cols, "Fatalities")

    # align with ForecastId explicitly
    submission = pd.DataFrame({
        "ForecastId": test["ForecastId"].values,
        "ConfirmedCases": conf_pred,
        "Fatalities": fat_pred,
    })

    submission = submission.sort_values("ForecastId").reset_index(drop=True)
    submission["ConfirmedCases"] = submission["ConfirmedCases"].clip(lower=0)
    submission["Fatalities"] = submission["Fatalities"].clip(lower=0)

    submission.to_csv(SUB_PATH, index=False)


if __name__ == "__main__":
    main()
