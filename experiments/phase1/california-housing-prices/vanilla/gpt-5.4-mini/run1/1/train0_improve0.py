

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

from sklearn.model_selection import train_test_split, KFold
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

# Keep the simpler imputation strategy
median_values = X.median(numeric_only=True)
X = X.fillna(median_values)
test = test.fillna(median_values)

# Preserve categorical features if they exist
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

# Small CV for more stable evaluation, while keeping runtime reasonable
use_cv = True
n_splits = 3

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

model_params = dict(
    iterations=10000,
    learning_rate=0.02,
    depth=8,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=200,
    l2_leaf_reg=6.0,
    random_strength=1.0,
    bagging_temperature=0.2,
    od_type="Iter",
    od_wait=200,
)

if use_cv:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols if len(cat_cols) > 0 else None)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols if len(cat_cols) > 0 else None)

        model = CatBoostRegressor(**model_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_fold_preds = model.predict(X_val)
        oof_preds[val_idx] = val_fold_preds
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_fold_preds))
        print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

        test_preds += model.predict(test) / n_splits

    cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"CV RMSE: {cv_rmse:.6f}")

else:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols if len(cat_cols) > 0 else None)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols if len(cat_cols) > 0 else None)

    model = CatBoostRegressor(**model_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_preds = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Final Validation Performance: {val_rmse:.6f}")

    test_preds = model.predict(test)

submission = pd.DataFrame({target_col: test_preds})
submission.to_csv("submission.csv", index=False)
