
import os
import sys
import subprocess
import math
import warnings

warnings.filterwarnings("ignore")


def ensure_package(package_name, import_name=None):
    __import__(import_name or package_name)


def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


try:
    ensure_package("catboost")
except ImportError:
    install_package("catboost")

try:
    ensure_package("lightgbm")
except ImportError:
    install_package("lightgbm")

import numpy as np
import pandas as pd
import torch

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cat_model = CatBoostRegressor(
        iterations=3000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        task_type="GPU" if device == "cuda" else "CPU",
    )

    lgb_model = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    if device == "cuda":
        lgb_model.set_params(device_type="gpu")

    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
    )

    cat_val_pred = cat_model.predict(X_val)
    lgb_val_pred = lgb_model.predict(X_val)

    cat_rmse = math.sqrt(mean_squared_error(y_val, cat_val_pred))
    lgb_rmse = math.sqrt(mean_squared_error(y_val, lgb_val_pred))

    cat_test_pred = cat_model.predict(test)
    lgb_test_pred = lgb_model.predict(test)

    ensemble_val_pred = 0.5 * cat_val_pred + 0.5 * lgb_val_pred
    ensemble_test_pred = 0.5 * cat_test_pred + 0.5 * lgb_test_pred

    ensemble_rmse = math.sqrt(mean_squared_error(y_val, ensemble_val_pred))

    print(f"CatBoost Validation Performance: {cat_rmse}")
    print(f"LightGBM Validation Performance: {lgb_rmse}")
    print(f"Final Validation Performance: {ensemble_rmse}")

    submission = pd.DataFrame({"median_house_value": ensemble_test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
