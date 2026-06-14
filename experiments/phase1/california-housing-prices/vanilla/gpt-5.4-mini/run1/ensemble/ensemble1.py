
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, Pool
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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

median_values = X.median(numeric_only=True)
X = X.fillna(median_values)
test = test.fillna(median_values)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

train_pool_a = Pool(X_tr, y_tr)
val_pool_a = Pool(X_val, y_val)

model_a = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=200,
)

model_a.fit(
    train_pool_a,
    eval_set=val_pool_a,
    use_best_model=True
)

pred_a_val = model_a.predict(X_val)
pred_a_test = model_a.predict(test)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof_b = np.zeros(len(X_tr), dtype=float)
test_b_folds = []

X_tr_arr = X_tr.reset_index(drop=True)
y_tr_arr = y_tr.reset_index(drop=True)
test_arr = test.reset_index(drop=True)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr_arr), 1):
    X_fold_tr = X_tr_arr.iloc[tr_idx]
    y_fold_tr = y_tr_arr.iloc[tr_idx]
    X_fold_va = X_tr_arr.iloc[va_idx]
    y_fold_va = y_tr_arr.iloc[va_idx]

    train_pool_b = Pool(X_fold_tr, y_fold_tr)
    val_pool_b = Pool(X_fold_va, y_fold_va)

    model_b = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        random_seed=SEED + fold,
        verbose=0,
    )

    model_b.fit(
        train_pool_b,
        eval_set=val_pool_b,
        use_best_model=True
    )

    oof_b[va_idx] = model_b.predict(X_fold_va)
    test_b_folds.append(model_b.predict(test_arr))

pred_b_test = np.mean(np.vstack(test_b_folds), axis=0)

def rank_calibrate(train_preds, train_target, test_preds):
    train_preds = np.asarray(train_preds, dtype=float)
    train_target = np.asarray(train_target, dtype=float)
    test_preds = np.asarray(test_preds, dtype=float)

    n = len(train_preds)
    if n < 2:
        mean_target = train_target.mean() if len(train_target) else 0.0
        return np.full_like(test_preds, mean_target, dtype=float), np.full_like(train_preds, mean_target, dtype=float)

    quantile_positions = np.linspace(0, 1, n)
    target_quantiles = np.quantile(train_target, quantile_positions)

    train_order = np.argsort(train_preds)
    train_u = np.empty(n, dtype=float)
    train_u[train_order] = np.linspace(0, 1, n)
    train_cal = np.interp(train_u, quantile_positions, target_quantiles)

    m = len(test_preds)
    test_order = np.argsort(test_preds)
    test_u = np.empty(m, dtype=float)
    test_u[test_order] = np.linspace(0, 1, m)
    test_cal = np.interp(test_u, quantile_positions, target_quantiles)

    return test_cal, train_cal

pred_a_test_cal, pred_a_tr_cal = rank_calibrate(pred_a_val, y_val.values, pred_a_test)
pred_b_test_cal, pred_b_oof_cal = rank_calibrate(oof_b, y_tr_arr.values, pred_b_test)

# Fix: only blend predictions for the same subset.
# For validation, use model A's validation predictions and model B's OOF predictions on the training split separately.
# Build a compatible hold-out validation blender using a simple mapping from model B via a fitted regressor on the training split.
val_blend_source_1 = pred_a_val
val_blend_source_2 = np.interp(
    np.linspace(0, 1, len(pred_a_val)),
    np.linspace(0, 1, len(pred_b_oof_cal)),
    np.sort(pred_b_oof_cal)
)

blended_val = np.median(np.vstack([val_blend_source_1, val_blend_source_2]), axis=0)

# For test blending, both arrays are aligned on the test set.
blended_test = np.median(np.vstack([pred_a_test_cal, pred_b_test_cal]), axis=0)

residual_model = LinearRegression()
residual_model.fit(blended_val.reshape(-1, 1), y_val.values)

final_val_preds = residual_model.predict(blended_val.reshape(-1, 1))
final_test_preds = residual_model.predict(blended_test.reshape(-1, 1))

final_val_preds = np.clip(final_val_preds, 0, None)
final_test_preds = np.clip(final_test_preds, 0, None)

val_rmse = np.sqrt(mean_squared_error(y_val, final_val_preds))

submission = pd.DataFrame({target_col: final_test_preds})
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {val_rmse}")
