
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

# Ensure required package is available
try:
    from catboost import CatBoostRegressor, Pool
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "median_house_value"
X = train.drop(columns=[target_col]).copy()
y = train[target_col].copy()

# -----------------------------
# Feature engineering
# -----------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Missingness indicators
na_cols = [c for c in X.columns if X[c].isna().any() or test[c].isna().any()]
for col in na_cols:
    X[f"{col}_isna"] = X[col].isna().astype(np.int8)
    test[f"{col}_isna"] = test[col].isna().astype(np.int8)

# Median imputation for original numeric columns
median_values = X[num_cols].median()
X[num_cols] = X[num_cols].fillna(median_values)
test[num_cols] = test[num_cols].fillna(median_values)

# Safe skew-aware transforms
numeric_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
skew_values = X[numeric_feature_cols].skew(numeric_only=True).replace([np.inf, -np.inf], np.nan)

for col in skew_values.index:
    if pd.notna(skew_values[col]) and abs(skew_values[col]) > 1.0:
        if (X[col] >= 0).all() and (test[col] >= 0).all():
            X[f"{col}_log1p"] = np.log1p(X[col].clip(lower=0))
            test[f"{col}_log1p"] = np.log1p(test[col].clip(lower=0))

# Ratios
common_ratio_pairs = [
    ("total_rooms", "households"),
    ("total_bedrooms", "total_rooms"),
    ("population", "households"),
    ("population", "total_rooms"),
]
for a, b in common_ratio_pairs:
    if a in X.columns and b in X.columns:
        X[f"{a}_per_{b}"] = X[a] / X[b].replace(0, np.nan)
        test[f"{a}_per_{b}"] = test[a] / test[b].replace(0, np.nan)

# Replace infinities and impute any new missing values
X = X.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)

all_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
fill_medians = X[all_num_cols].median()
X[all_num_cols] = X[all_num_cols].fillna(fill_medians)
test[all_num_cols] = test[all_num_cols].fillna(fill_medians)

# -----------------------------
# Deterministic KFold OOF + test ensembling
# -----------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

seeds = [SEED, SEED + 7, SEED + 21]
oof_pred = np.zeros(len(X), dtype=float)
test_pred = np.zeros(len(test), dtype=float)

model_params = dict(
    iterations=5000,
    learning_rate=0.025,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    bagging_temperature=0.4,
    l2_leaf_reg=5.0,
    min_data_in_leaf=20,
    random_strength=0.8,
    od_type="Iter",
    od_wait=200,
    allow_writing_files=False,
    verbose=200,
)

# Average OOF and test predictions correctly across seeds and folds
for seed in seeds:
    fold_test_pred = np.zeros(len(test), dtype=float)
    fold_oof_pred = np.zeros(len(X), dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)

        params = model_params.copy()
        params["random_seed"] = seed + fold

        model = CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        val_preds = model.predict(X_val)
        fold_oof_pred[val_idx] = val_preds
        fold_test_pred += model.predict(test) / n_splits

    oof_pred += fold_oof_pred / len(seeds)
    test_pred += fold_test_pred / len(seeds)

final_validation_score = np.sqrt(mean_squared_error(y, oof_pred))

# Submission must contain the target column only, per provided format
submission = pd.DataFrame({target_col: test_pred})
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
print(submission.head())
