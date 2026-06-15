
import sys
import subprocess
import math
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])


ensure_package("lightgbm")
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
    params = dict(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model = LGBMRegressor(**params)

    if device == "cuda":
        model.set_params(device_type="gpu")

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
    )

    val_pred = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Final Validation Performance: {rmse}")

    test_pred = model.predict(test)
    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
