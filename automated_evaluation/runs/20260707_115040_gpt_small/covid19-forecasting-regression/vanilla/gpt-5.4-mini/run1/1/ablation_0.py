
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


def prepare_data(train):
    if "Province_State" not in train.columns:
        train["Province_State"] = ""
    train["Province_State"] = train["Province_State"].fillna("").astype(str)
    train["Country_Region"] = train["Country_Region"].fillna("").astype(str)
    train["Date"] = pd.to_datetime(train["Date"])

    train["province_missing"] = (train["Province_State"].str.strip() == "").astype(int)
    train["country_missing"] = (train["Country_Region"].str.strip() == "").astype(int)

    combined = train[["Province_State", "Country_Region"]].copy()
    combined["region_key"] = combined["Country_Region"] + "||" + combined["Province_State"]
    region_counts = combined["region_key"].value_counts()
    country_counts = combined["Country_Region"].value_counts()

    min_date = train["Date"].min()
    train["region_key"] = train["Country_Region"] + "||" + train["Province_State"]
    train["region_count"] = train["region_key"].map(region_counts).fillna(0).astype(int)
    train["country_count"] = train["Country_Region"].map(country_counts).fillna(0).astype(int)
    train["region_freq"] = train["region_count"]
    train["country_freq"] = train["country_count"]
    add_date_features(train)
    train["days_since_start"] = (train["Date"] - min_date).dt.days
    train["log_region_count"] = np.log1p(train["region_count"])
    train["log_country_count"] = np.log1p(train["country_count"])
    return train


def encode_categoricals(train):
    for c in ["Province_State", "Country_Region"]:
        le = LabelEncoder()
        le.fit(train[c].astype(str))
        train[c] = le.transform(train[c].astype(str))
    return train


def make_holdout_split(df, val_days=14):
    max_date = df["Date"].max()
    split_date = max_date - pd.Timedelta(days=val_days)
    tr_idx = df["Date"] <= split_date
    va_idx = df["Date"] > split_date
    return tr_idx, va_idx


def hgb_train_predict(train_part, valid_part, feature_cols, target_col, use_log_features=True):
    cols = feature_cols.copy()
    if not use_log_features:
        cols = [c for c in cols if c not in ["log_region_count", "log_country_count"]]

    X_train = train_part[cols]
    y_train = np.log1p(train_part[target_col].clip(lower=0))
    X_valid = valid_part[cols]

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
    return np.maximum(valid_pred, 0)


def cat_train_predict(train_part, valid_part, feature_cols, target_col, use_log_features=True, use_date_features=True):
    cols = feature_cols.copy()
    if not use_log_features:
        cols = [c for c in cols if c not in ["log_region_count", "log_country_count"]]
    if not use_date_features:
        cols = [c for c in cols if c not in ["year", "month", "day", "dow", "dayofyear", "weekofyear", "is_month_start", "is_month_end", "days_since_start"]]

    X_train = train_part[cols]
    y_train = np.log1p(train_part[target_col].clip(lower=0))
    X_valid = valid_part[cols]

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
    return np.maximum(valid_pred, 0)


def eval_setting(train_part, valid_part, feature_cols, target, ablation_name,
                 use_log_features=True, use_date_features=True, disable_cat=False, disable_hgb=False):
    preds = []

    if not disable_hgb:
        hgb_valid = hgb_train_predict(train_part, valid_part, feature_cols, target, use_log_features=use_log_features)
        preds.append(hgb_valid)

    if not disable_cat:
        cat_valid = cat_train_predict(
            train_part, valid_part, feature_cols, target,
            use_log_features=use_log_features,
            use_date_features=use_date_features
        )
        preds.append(cat_valid)

    if len(preds) == 1:
        pred = preds[0]
    else:
        pred = np.mean(np.vstack(preds), axis=0)

    score = rmsle(valid_part[target], pred)
    print(f"{ablation_name} | {target} RMSLE: {score:.6f}")
    return score


def main():
    train = pd.read_csv(TRAIN_PATH)

    if len(train) > SUBSAMPLE_MAX_ROWS:
        train = train.sample(SUBSAMPLE_MAX_ROWS, random_state=RANDOM_STATE).copy()

    train = prepare_data(train)
    train = encode_categoricals(train)
    train = train.sort_values(["Country_Region", "Province_State", "Date"]).reset_index(drop=True)

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

    tr_idx, va_idx = make_holdout_split(train, val_days=VALIDATION_DAYS)
    train_part = train.loc[tr_idx].copy()
    valid_part = train.loc[va_idx].copy()

    experiments = [
        ("Baseline ensemble", dict(use_log_features=True, use_date_features=True, disable_cat=False, disable_hgb=False)),
        ("Ablation 1: remove log count features", dict(use_log_features=False, use_date_features=True, disable_cat=False, disable_hgb=False)),
        ("Ablation 2: remove date features from CatBoost", dict(use_log_features=True, use_date_features=False, disable_cat=False, disable_hgb=False)),
        ("Ablation 3: HGB only", dict(use_log_features=True, use_date_features=True, disable_cat=False, disable_hgb=True)),
        ("Ablation 4: CatBoost only", dict(use_log_features=True, use_date_features=True, disable_cat=True, disable_hgb=False)),
    ]

    results = []
    for name, kwargs in experiments:
        case_score = eval_setting(train_part, valid_part, feature_cols, "ConfirmedCases", name, **kwargs)
        fatal_score = eval_setting(train_part, valid_part, feature_cols, "Fatalities", name, **kwargs)
        mean_score = (case_score + fatal_score) / 2.0
        results.append((name, mean_score))
        print(f"{name} | Mean Validation RMSLE: {mean_score:.6f}\n")

    baseline = results[0][1]
    print("Ablation summary vs baseline:")
    best_drop = None
    best_name = None

    for name, score in results[1:]:
        delta = score - baseline
        print(f"{name}: {score:.6f} | Delta: {delta:+.6f}")
        if best_drop is None or delta > best_drop:
            best_drop = delta
            best_name = name

    print("\nMost important component (largest performance drop when removed):")
    print(f"{best_name} | Baseline Mean RMSLE: {baseline:.6f} -> Ablated Mean RMSLE: {baseline + best_drop:.6f} | Drop: {best_drop:+.6f}")


if __name__ == "__main__":
    main()
