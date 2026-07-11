
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

INPUT_DIR = "./input"
FINAL_DIR = "./final"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

os.makedirs(FINAL_DIR, exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df = df.copy()

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["weekday"] = dt.dt.weekday
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["weekend"] = (df["weekday"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["hour_x_workingday"] = df["hour"] * df["workingday"]
    df["hour_x_season"] = df["hour"] * df["season"]
    df["hour_x_weekend"] = df["hour"] * df["weekend"]
    df["temp_x_humidity"] = df["temp"] * df["humidity"]
    df["temp_x_windspeed"] = df["temp"] * df["windspeed"]

    return df


train = add_datetime_features(train)
test = add_datetime_features(test)

features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "year", "weekday",
    "dayofyear", "weekofyear", "weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos",
    "hour_x_workingday", "hour_x_season", "hour_x_weekend",
    "temp_x_humidity", "temp_x_windspeed"
]

train = train.sort_values("datetime").reset_index(drop=True)

n_folds = 5
fold_sizes = np.full(n_folds, len(train) // n_folds, dtype=int)
fold_sizes[: len(train) % n_folds] += 1
fold_boundaries = np.cumsum(fold_sizes)

oof_lgbm = np.zeros(len(train), dtype=float)
oof_hgb = np.zeros(len(train), dtype=float)

start = 0
for end in fold_boundaries:
    valid_idx = np.arange(start, end)
    train_idx = np.arange(0, start)

    if len(train_idx) < 100:
        start = end
        continue

    X_train = train.loc[train_idx, features]
    y_train = np.log1p(train.loc[train_idx, "count"].astype(float))
    X_valid = train.loc[valid_idx, features]

    lgbm_model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    hgb_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False
    )

    lgbm_model.fit(X_train, y_train)
    hgb_model.fit(X_train, y_train)

    oof_lgbm[valid_idx] = np.clip(np.expm1(lgbm_model.predict(X_valid)), 0, None)
    oof_hgb[valid_idx] = np.clip(np.expm1(hgb_model.predict(X_valid)), 0, None)

    start = end

covered = oof_lgbm > 0
if not np.all(covered):
    first_valid_start = np.argmax(covered)
    if first_valid_start > 0:
        X_train = train.loc[:first_valid_start - 1, features]
        y_train = np.log1p(train.loc[:first_valid_start - 1, "count"].astype(float))
        X_valid = train.loc[first_valid_start:, features]

        lgbm_model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        hgb_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            l2_regularization=1.0,
            random_state=42,
            early_stopping=False
        )
        lgbm_model.fit(X_train, y_train)
        hgb_model.fit(X_train, y_train)

        oof_lgbm[first_valid_start:] = np.clip(np.expm1(lgbm_model.predict(X_valid)), 0, None)
        oof_hgb[first_valid_start:] = np.clip(np.expm1(hgb_model.predict(X_valid)), 0, None)

meta_X = np.column_stack([
    np.log1p(np.clip(oof_lgbm, 0, None)),
    np.log1p(np.clip(oof_hgb, 0, None))
])
meta_y = np.log1p(train["count"].astype(float).values)

meta_model = Ridge(alpha=1e-3, fit_intercept=True, random_state=42)
meta_model.fit(meta_X, meta_y)

X_full = train[features]
y_full = np.log1p(train["count"].astype(float))

lgbm_full = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

hgb_full = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=500,
    l2_regularization=1.0,
    random_state=42,
    early_stopping=False
)

lgbm_full.fit(X_full, y_full)
hgb_full.fit(X_full, y_full)

test_pred_lgbm = np.clip(np.expm1(lgbm_full.predict(test[features])), 0, None)
test_pred_hgb = np.clip(np.expm1(hgb_full.predict(test[features])), 0, None)

test_meta_X = np.column_stack([
    np.log1p(test_pred_lgbm),
    np.log1p(test_pred_hgb)
])
test_pred = np.expm1(meta_model.predict(test_meta_X))
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "datetime": test["datetime"],
    "count": test_pred
})
submission.to_csv(os.path.join(FINAL_DIR, "submission.csv"), index=False)

print(submission.head())
