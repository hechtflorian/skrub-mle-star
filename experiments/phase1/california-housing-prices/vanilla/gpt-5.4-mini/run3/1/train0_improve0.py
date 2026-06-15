
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

    model = CatBoostRegressor(
        iterations=5000,
        depth=9,
        learning_rate=0.025,
        loss_function="RMSE",
        random_seed=42,
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
    mse = mean_squared_error(y_val, val_pred)
    rmse = math.sqrt(mse)
    print(f"Final Validation Performance: {rmse}")

    test_pred = model.predict(test)
    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
