
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_and_predict(X_tr, X_val, y_tr, y_val, test, seed, depth, learning_rate, iterations):
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
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
    test_pred = model.predict(test)
    rmse = math.sqrt(mean_squared_error(y_val, val_pred))
    return rmse, test_pred

def main():
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop(columns=["median_house_value"])
    y = train["median_house_value"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Stream A: original-style CatBoost
    rmse_a, pred_a = train_and_predict(
        X_tr, X_val, y_tr, y_val, test,
        seed=42, depth=9, learning_rate=0.025, iterations=5000
    )

    # Stream B: slightly different CatBoost variant for diversity
    rmse_b, pred_b = train_and_predict(
        X_tr, X_val, y_tr, y_val, test,
        seed=2024, depth=8, learning_rate=0.03, iterations=5000
    )

    # Weighted average based on holdout RMSE (lower RMSE gets higher weight)
    weight_a = 1.0 / (rmse_a + 1e-12)
    weight_b = 1.0 / (rmse_b + 1e-12)
    final_test_pred = (weight_a * pred_a + weight_b * pred_b) / (weight_a + weight_b)

    # Compute ensemble validation performance using holdout predictions
    # Refit validation predictions via weighted average on val set
    model_a = CatBoostRegressor(
        iterations=5000,
        depth=9,
        learning_rate=0.025,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "CPU"
    )
    model_b = CatBoostRegressor(
        iterations=5000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=2024,
        verbose=200,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "CPU"
    )

    model_a.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=200
    )
    model_b.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=200
    )

    val_pred_a = model_a.predict(X_val)
    val_pred_b = model_b.predict(X_val)
    final_val_pred = (weight_a * val_pred_a + weight_b * val_pred_b) / (weight_a + weight_b)

    mse = mean_squared_error(y_val, final_val_pred)
    rmse = math.sqrt(mse)
    print(f"Final Validation Performance: {rmse}")

    submission = pd.DataFrame({"median_house_value": final_test_pred})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
