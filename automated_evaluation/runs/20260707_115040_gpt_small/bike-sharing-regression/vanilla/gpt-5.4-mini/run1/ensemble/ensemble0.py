
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

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

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Interaction features
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

# Time-based CV backbone
n_folds = 5
fold_size = len(train) // (n_folds + 1)
folds = []
for i in range(n_folds):
    train_end = fold_size * (i + 1)
    valid_end = fold_size * (i + 2) if i < n_folds - 1 else len(train)
    tr_idx = np.arange(0, train_end)
    va_idx = np.arange(train_end, valid_end)
    if len(va_idx) > 0 and len(tr_idx) > 0:
        folds.append((tr_idx, va_idx))

oof_lgbm = np.zeros(len(train), dtype=float)
oof_hgb = np.zeros(len(train), dtype=float)

test_pred_lgbm_folds = []
test_pred_hgb_folds = []

for fold_id, (tr_idx, va_idx) in enumerate(folds):
    train_part = train.iloc[tr_idx].copy()
    valid_part = train.iloc[va_idx].copy()

    X_train = train_part[features]
    X_valid = valid_part[features]

    y_train = np.log1p(train_part["count"].astype(float))
    y_valid = valid_part["count"].astype(float)

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

    valid_pred_lgbm = np.expm1(lgbm_model.predict(X_valid))
    valid_pred_hgb = np.expm1(hgb_model.predict(X_valid))

    oof_lgbm[va_idx] = np.clip(valid_pred_lgbm, 0, None)
    oof_hgb[va_idx] = np.clip(valid_pred_hgb, 0, None)

    test_pred_lgbm_folds.append(np.expm1(lgbm_model.predict(test[features])))
    test_pred_hgb_folds.append(np.expm1(hgb_model.predict(test[features])))

# Global blend weight search on full OOF vectors
y_true = train["count"].astype(float).values
blend_candidates = np.arange(0.0, 1.0001, 0.05)
best_w = 0.6
best_score = np.inf

for w in blend_candidates:
    oof_pred = w * oof_lgbm + (1 - w) * oof_hgb
    oof_pred = np.clip(oof_pred, 0, None)
    score = np.sqrt(mean_squared_log_error(y_true, oof_pred))
    if score < best_score:
        best_score = score
        best_w = w

print(f"Best global blend weight for LGBM: {best_w:.2f}")
print(f"Final Validation Performance: {best_score}")

# Refit both models on full training data
X_full = train[features]
y_full = np.log1p(train["count"].astype(float))

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

lgbm_model.fit(X_full, y_full)
hgb_model.fit(X_full, y_full)

test_pred_lgbm = np.expm1(lgbm_model.predict(test[features]))
test_pred_hgb = np.expm1(hgb_model.predict(test[features]))

test_pred = best_w * test_pred_lgbm + (1 - best_w) * test_pred_hgb
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "datetime": test["datetime"],
    "count": test_pred
})
submission.to_csv("submission.csv", index=False)

print(submission.head())
