import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

train = pd.read_csv(TRAIN_PATH)

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
    return df

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

train = add_datetime_features(train)
train = train.sort_values("datetime").reset_index(drop=True)

features_base = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "year", "weekday",
    "dayofyear", "weekofyear", "weekend"
]

split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

X_train_base = train_part[features_base]
y_train = train_part["count"].astype(float)
X_valid_base = valid_part[features_base]
y_valid = valid_part["count"].astype(float)

def evaluate_model(X_tr, X_va, description):
    lgbm = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False
    )
    lgbm.fit(X_tr, y_train)
    hgb.fit(X_tr, y_train)

    pred_lgbm = np.clip(lgbm.predict(X_va), 0, None)
    pred_hgb = np.clip(hgb.predict(X_va), 0, None)
    pred = 0.6 * pred_lgbm + 0.4 * pred_hgb
    score = rmsle(y_valid, pred)
    print(f"{description}: RMSLE = {score:.6f}")
    return score

# Baseline
baseline_score = evaluate_model(X_train_base, X_valid_base, "Baseline")

# Ablation 1: remove engineered datetime features
features_no_datetime = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed"
]
X_train_no_datetime = train_part[features_no_datetime]
X_valid_no_datetime = valid_part[features_no_datetime]
score_no_datetime = evaluate_model(X_train_no_datetime, X_valid_no_datetime, "Ablation 1 - No datetime features")

# Ablation 2: disable model ensembling by using only LightGBM
def evaluate_lgbm_only(X_tr, X_va, description):
    lgbm = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    lgbm.fit(X_tr, y_train)
    pred = np.clip(lgbm.predict(X_va), 0, None)
    score = rmsle(y_valid, pred)
    print(f"{description}: RMSLE = {score:.6f}")
    return score

score_lgbm_only = evaluate_lgbm_only(X_train_base, X_valid_base, "Ablation 2 - LightGBM only")

# Compare impacts
results = {
    "Baseline": baseline_score,
    "No datetime features": score_no_datetime,
    "LightGBM only": score_lgbm_only
}

best_ablation = max(results, key=lambda k: results[k] - baseline_score)
worst_ablation = max(results, key=lambda k: baseline_score - results[k])

print("\nPerformance impact vs baseline:")
for name, score in results.items():
    delta = score - baseline_score
    print(f"{name}: delta RMSLE = {delta:+.6f}")

print(f"\nMost important part contributing to performance: {worst_ablation}")