
import os
import sys
import subprocess
import math

def ensure_package(package_name, import_name=None):
    try:
        __import__(import_name or package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])

ensure_package("catboost")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def train_and_predict_fold(X_tr, y_tr, X_val, y_val, X_test, target_clip_low, target_clip_high, seed):
    model = CatBoostRegressor(
        iterations=5000,
        depth=9,
        learning_rate=0.025,
        loss_function="RMSE",
        random_seed=seed,
        verbose=200,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "CPU"
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=200
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_pred = np.clip(val_pred, target_clip_low, target_clip_high)
    test_pred = np.clip(test_pred, target_clip_low, target_clip_high)

    return val_pred, test_pred

def main():
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop(columns=["median_house_value"])
    y = train["median_house_value"].values

    target_clip_low = np.quantile(y, 0.01)
    target_clip_high = np.quantile(y, 0.99)

    # Hold-out validation set for evaluation metric reporting
    kf_holdout = KFold(n_splits=5, shuffle=True, random_state=42)
    holdout_train_idx, holdout_val_idx = next(kf_holdout.split(X))

    X_hold_tr = X.iloc[holdout_train_idx]
    y_hold_tr = y[holdout_train_idx]
    X_hold_val = X.iloc[holdout_val_idx]
    y_hold_val = y[holdout_val_idx]

    # Train a single hold-out model to compute validation metric
    hold_model = CatBoostRegressor(
        iterations=5000,
        depth=9,
        learning_rate=0.025,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "CPU"
    )
    hold_model.fit(
        X_hold_tr,
        y_hold_tr,
        eval_set=(X_hold_val, y_hold_val),
        use_best_model=True,
        early_stopping_rounds=200
    )
    hold_val_pred = hold_model.predict(X_hold_val)
    hold_val_pred = np.clip(hold_val_pred, target_clip_low, target_clip_high)
    mse = mean_squared_error(y_hold_val, hold_val_pred)
    rmse = math.sqrt(mse)
    print(f"Final Validation Performance: {rmse}")

    # Fold-based ensemble for test prediction: median of clipped fold predictions
    fold_test_preds_seed42 = []
    kf1 = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf1.split(X), 1):
        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]

        _, test_pred = train_and_predict_fold(
            X_tr, y_tr, X_val, y_val, test,
            target_clip_low, target_clip_high,
            seed=42 + fold
        )
        fold_test_preds_seed42.append(test_pred)

    fold_test_preds_seed7 = []
    kf2 = KFold(n_splits=5, shuffle=True, random_state=7)

    for fold, (tr_idx, val_idx) in enumerate(kf2.split(X), 1):
        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]

        _, test_pred = train_and_predict_fold(
            X_tr, y_tr, X_val, y_val, test,
            target_clip_low, target_clip_high,
            seed=700 + fold
        )
        fold_test_preds_seed7.append(test_pred)

    fold_test_preds_seed42 = np.stack(fold_test_preds_seed42, axis=0)
    fold_test_preds_seed7 = np.stack(fold_test_preds_seed7, axis=0)

    median_pred_42 = np.median(fold_test_preds_seed42, axis=0)
    median_pred_7 = np.median(fold_test_preds_seed7, axis=0)

    final_test_pred = (median_pred_42 + median_pred_7) / 2.0
    final_test_pred = np.clip(final_test_pred, target_clip_low, target_clip_high)

    submission = pd.DataFrame({"median_house_value": final_test_pred})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
