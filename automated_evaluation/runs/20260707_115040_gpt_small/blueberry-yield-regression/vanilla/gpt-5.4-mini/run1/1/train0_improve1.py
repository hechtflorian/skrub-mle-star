
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

df = pd.read_csv(train_path)

X = df.drop(columns=["id", "yield"])
y = df["yield"]

test = pd.read_csv(test_path)
X_test = test.drop(columns=["id"])

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Robust binning for stratification; fallback to KFold if stratification is not feasible
n_splits = 5
use_stratified = True
try:
    # Create bins with fewer quantiles if needed and ensure enough unique classes
    n_bins = min(10, len(y_tr))
    y_bins = pd.qcut(y_tr, q=n_bins, labels=False, duplicates="drop")
    if y_bins.nunique() < n_splits:
        use_stratified = False
except Exception:
    use_stratified = False

oof_pred = np.zeros(len(y_tr))
test_pred = np.zeros(len(X_test))

if use_stratified:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    split_iter = splitter.split(X_tr, y_bins)
else:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    split_iter = splitter.split(X_tr)

for fold, split_data in enumerate(split_iter, 1):
    if use_stratified:
        tr_idx, va_idx = split_data
    else:
        tr_idx, va_idx = split_data

    X_train_fold, X_valid_fold = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    y_train_fold, y_valid_fold = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.03,
        depth=5,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=8.0,
        bagging_temperature=0.5,
        subsample=0.8,
        random_seed=SEED + fold,
        verbose=200,
        allow_writing_files=False
    )

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_valid_fold, y_valid_fold),
        use_best_model=True
    )

    oof_pred[va_idx] = model.predict(X_valid_fold)
    test_pred += model.predict(X_test) / n_splits

oof_mae = mean_absolute_error(y_tr, oof_pred)
final_validation_score = oof_mae
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission.csv", index=False)
