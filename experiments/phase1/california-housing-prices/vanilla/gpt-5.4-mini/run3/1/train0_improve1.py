
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
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

def main():
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop(columns=["median_house_value"])
    y = train["median_house_value"]

    # Holdout split for final validation print
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_features = []  # all features are numeric in this dataset

    params = dict(
        iterations=3000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "CPU",
        subsample=0.8,
        rsm=0.8,
        l2_leaf_reg=5,
        min_data_in_leaf=20,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_tr))
    test_pred = np.zeros(len(test))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr, y_tr), 1):
        X_train_fold, X_valid_fold = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_train_fold, y_valid_fold = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]

        train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_features)
        valid_pool = Pool(X_valid_fold, y_valid_fold, cat_features=cat_features)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=100)

        oof_pred[val_idx] = model.predict(valid_pool)
        models.append(model)

        test_pred += model.predict(test) / kf.n_splits

    # Final validation performance on the holdout split
    val_pred = np.mean([m.predict(X_val) for m in models], axis=0)
    rmse = math.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Final Validation Performance: {rmse}")

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
