
import os
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

    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_quarter_start"] = dt.dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

    weekday_peak = df["hour"].isin([7, 8, 9, 17, 18, 19])
    weekend_peak = df["hour"].isin([10, 11, 12, 13, 14, 15, 16, 17])
    df["is_peak_hour"] = (
        ((df["workingday"] == 1) & weekday_peak) |
        ((df["workingday"] == 0) & weekend_peak)
    ).astype(int)

    return df

def safe_msle(y_true, y_pred):
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return mean_squared_log_error(y_true, y_pred)

train = add_datetime_features(train)
test = add_datetime_features(test)

features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "year", "weekday",
    "dayofyear", "weekofyear", "weekend",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_peak_hour"
]

train = train.sort_values("datetime").reset_index(drop=True)
test = test.sort_values("datetime").reset_index(drop=True)

n_splits = 4
min_train_size = max(200, int(len(train) * 0.5))
fold_size = max((len(train) - min_train_size) // (n_splits + 1), 1)

candidate_configs = [
    {"weight": 0.6, "lgb_log1p": False, "hgb_log1p": False},
    {"weight": 0.5, "lgb_log1p": True,  "hgb_log1p": False},
    {"weight": 0.5, "lgb_log1p": False, "hgb_log1p": True},
    {"weight": 0.7, "lgb_log1p": True,  "hgb_log1p": False},
    {"weight": 0.7, "lgb_log1p": False, "hgb_log1p": True},
]

oof_results = []
best_config = None
best_score = np.inf

for cfg in candidate_configs:
    fold_scores = []
    oof_pred = np.zeros(len(train), dtype=float)
    oof_mask = np.zeros(len(train), dtype=bool)

    for fold in range(n_splits):
        valid_start = min_train_size + fold * fold_size
        valid_end = min(valid_start + fold_size, len(train))
        if valid_start >= len(train) or valid_start >= valid_end:
            continue

        train_part = train.iloc[:valid_start].copy()
        valid_part = train.iloc[valid_start:valid_end].copy()

        X_train = train_part[features]
        y_train = train_part["count"].astype(float)
        X_valid = valid_part[features]
        y_valid = valid_part["count"].astype(float)

        y_train_lgb = np.log1p(y_train) if cfg["lgb_log1p"] else y_train
        y_train_hgb = np.log1p(y_train) if cfg["hgb_log1p"] else y_train

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

        lgbm_model.fit(X_train, y_train_lgb)
        hgb_model.fit(X_train, y_train_hgb)

        valid_pred_lgbm = lgbm_model.predict(X_valid)
        valid_pred_hgb = hgb_model.predict(X_valid)

        if cfg["lgb_log1p"]:
            valid_pred_lgbm = np.expm1(valid_pred_lgbm)
        if cfg["hgb_log1p"]:
            valid_pred_hgb = np.expm1(valid_pred_hgb)

        valid_pred_lgbm = np.clip(valid_pred_lgbm, 0, None)
        valid_pred_hgb = np.clip(valid_pred_hgb, 0, None)
        valid_pred = np.clip(cfg["weight"] * valid_pred_lgbm + (1.0 - cfg["weight"]) * valid_pred_hgb, 0, None)

        fold_score = np.sqrt(safe_msle(y_valid, valid_pred))
        fold_scores.append(fold_score)

        oof_pred[valid_start:valid_end] = valid_pred
        oof_mask[valid_start:valid_end] = True

    if len(fold_scores) == 0:
        continue

    avg_score = float(np.mean(fold_scores))
    oof_score = np.sqrt(safe_msle(
        train.loc[oof_mask, "count"].astype(float),
        np.clip(oof_pred[oof_mask], 0, None)
    ))

    oof_results.append({
        "config": cfg,
        "fold_scores": fold_scores,
        "avg_score": avg_score,
        "oof_score": oof_score
    })

    if avg_score < best_score:
        best_score = avg_score
        best_config = cfg

print("CV results:")
for res in oof_results:
    print(
        res["config"],
        "folds=", np.round(res["fold_scores"], 5).tolist(),
        "avg=", round(res["avg_score"], 5),
        "oof=", round(res["oof_score"], 5)
    )

print(f"Selected config: {best_config}, CV score: {best_score}")

X_full = train[features]
y_full = train["count"].astype(float)

y_full_lgb = np.log1p(y_full) if best_config["lgb_log1p"] else y_full
y_full_hgb = np.log1p(y_full) if best_config["hgb_log1p"] else y_full

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

lgbm_model.fit(X_full, y_full_lgb)
hgb_model.fit(X_full, y_full_hgb)

test_pred_lgbm = lgbm_model.predict(test[features])
test_pred_hgb = hgb_model.predict(test[features])

if best_config["lgb_log1p"]:
    test_pred_lgbm = np.expm1(test_pred_lgbm)
if best_config["hgb_log1p"]:
    test_pred_hgb = np.expm1(test_pred_hgb)

test_pred_lgbm = np.clip(test_pred_lgbm, 0, None)
test_pred_hgb = np.clip(test_pred_hgb, 0, None)
test_pred = np.clip(best_config["weight"] * test_pred_lgbm + (1.0 - best_config["weight"]) * test_pred_hgb, 0, None)

best_oof_pred = np.zeros(len(train), dtype=float)
best_oof_mask = np.zeros(len(train), dtype=bool)

for fold in range(n_splits):
    valid_start = min_train_size + fold * fold_size
    valid_end = min(valid_start + fold_size, len(train))
    if valid_start >= len(train) or valid_start >= valid_end:
        continue

    train_part = train.iloc[:valid_start].copy()
    valid_part = train.iloc[valid_start:valid_end].copy()

    X_train = train_part[features]
    y_train = train_part["count"].astype(float)
    X_valid = valid_part[features]

    y_train_lgb = np.log1p(y_train) if best_config["lgb_log1p"] else y_train
    y_train_hgb = np.log1p(y_train) if best_config["hgb_log1p"] else y_train

    lgbm_model_cv = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    hgb_model_cv = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False
    )

    lgbm_model_cv.fit(X_train, y_train_lgb)
    hgb_model_cv.fit(X_train, y_train_hgb)

    p1 = lgbm_model_cv.predict(X_valid)
    p2 = hgb_model_cv.predict(X_valid)

    if best_config["lgb_log1p"]:
        p1 = np.expm1(p1)
    if best_config["hgb_log1p"]:
        p2 = np.expm1(p2)

    p = np.clip(
        best_config["weight"] * np.clip(p1, 0, None) + (1.0 - best_config["weight"]) * np.clip(p2, 0, None),
        0,
        None
    )
    best_oof_pred[valid_start:valid_end] = p
    best_oof_mask[valid_start:valid_end] = True

train_mean = float(np.mean(train.loc[best_oof_mask, "count"].astype(float)))
oof_mean = float(np.mean(best_oof_pred[best_oof_mask])) if np.any(best_oof_mask) else train_mean
if oof_mean > 0:
    test_pred *= train_mean / oof_mean

final_validation_score = float(np.sqrt(safe_msle(
    train.loc[best_oof_mask, "count"].astype(float),
    np.clip(best_oof_pred[best_oof_mask], 0, None)
)))
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    "datetime": test["datetime"],
    "count": np.clip(test_pred, 0, None)
})
submission.to_csv("submission.csv", index=False)

print(submission.head())
