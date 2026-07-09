
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
    # Kept for backward compatibility, but CatBoost is the primary branch per the improvement plan.
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


def _build_cat_features(train_df, test_df):
    # Richer temporal encodings and simple trend proxies, while preserving the existing date features.
    for df in (train_df, test_df):
        date = pd.to_datetime(df["Date"])
        df["year"] = date.dt.year
        df["month"] = date.dt.month
        df["day"] = date.dt.day
        df["dow"] = date.dt.dayofweek
        df["dayofyear"] = date.dt.dayofyear
        df["weekofyear"] = date.dt.isocalendar().week.astype(int)
        df["is_month_start"] = date.dt.is_month_start.astype(int)
        df["is_month_end"] = date.dt.is_month_end.astype(int)
        df["days_since_start"] = (date - date.min()).dt.days.astype(int)

        # Cyclical encodings for seasonality
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
        df["quarter"] = date.dt.quarter
        df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Trend proxy features from the training history only
    key_cols = ["Country_Region", "Province_State"]
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["_is_train"] = 1
    test_df["_is_train"] = 0
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)

    combined = combined.sort_values(key_cols + ["Date"]).reset_index(drop=True)
    for group_col in key_cols:
        combined[f"{group_col.lower()}_day_index"] = combined.groupby(group_col).cumcount()

    # simple within-region cumulative counts over time as a proxy for trend/age
    combined["region_time_rank"] = combined.groupby(key_cols)["days_since_start"].rank(method="first")
    combined["region_time_rank"] = combined["region_time_rank"].fillna(0)

    train_out = combined[combined["_is_train"] == 1].drop(columns=["_is_train"]).copy()
    test_out = combined[combined["_is_train"] == 0].drop(columns=["_is_train"]).copy()
    return train_out, test_out


def cat_train_predict(train_part, valid_part, test_df, feature_cols, target_col):
    X_train = train_part[feature_cols]
    y_train = np.log1p(train_part[target_col].clip(lower=0))
    X_valid = valid_part[feature_cols]
    X_test = test_df[feature_cols]

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=2200,
        depth=7,
        learning_rate=0.035,
        l2_leaf_reg=5.0,
        subsample=0.85,
        colsample_bylevel=0.85,
        random_strength=1.0,
        bagging_temperature=0.5,
        min_data_in_leaf=20,
        verbose=False,
        random_seed=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    valid_pred = np.expm1(model.predict(X_valid))
    test_pred = np.expm1(model.predict(X_test))
    return np.maximum(valid_pred, 0), np.maximum(test_pred, 0)


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
        # richer temporal encodings / trend proxies
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "quarter",
        "is_weekend",
        "country_region_day_index",
        "province_state_day_index",
        "region_time_rank",
    ]

    train, test = _build_cat_features(train, test)

    for c in feature_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(0)
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0)

    tr_idx, va_idx = make_holdout_split(train, val_days=VALIDATION_DAYS)
    train_part = train.loc[tr_idx].copy()
    valid_part = train.loc[va_idx].copy()

    valid_pred = pd.DataFrame(index=valid_part.index)
    test_pred = pd.DataFrame(index=test.index)

    for target in ["ConfirmedCases", "Fatalities"]:
        # CatBoost-only pipeline per the ablation result
        cat_valid, cat_test = cat_train_predict(train_part, valid_part, test, feature_cols, target)
        valid_pred[target] = cat_valid
        test_pred[target] = cat_test

    y_true_valid = valid_part[["ConfirmedCases", "Fatalities"]].copy()
    y_pred_valid = valid_pred[["ConfirmedCases", "Fatalities"]].copy()

    rmsle_cases = rmsle(y_true_valid["ConfirmedCases"], y_pred_valid["ConfirmedCases"])
    rmsle_fatal = rmsle(y_true_valid["Fatalities"], y_pred_valid["Fatalities"])
    final_validation_score = float((rmsle_cases + rmsle_fatal) / 2.0)

    submission = pd.DataFrame({
        "ForecastId": test["ForecastId"].values,
        "ConfirmedCases": test_pred["ConfirmedCases"].values,
        "Fatalities": test_pred["Fatalities"].values,
    })
    submission = submission.sort_values("ForecastId").reset_index(drop=True)
    submission["ConfirmedCases"] = submission["ConfirmedCases"].clip(lower=0)
    submission["Fatalities"] = submission["Fatalities"].clip(lower=0)
    submission.to_csv(SUB_PATH, index=False)

    print(f"Final Validation Performance: {final_validation_score}")



if __name__ == "__main__":
    main()
