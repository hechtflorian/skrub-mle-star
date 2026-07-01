
import os
import warnings
import subprocess
import sys

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub"])
    import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PowerTransformer


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def main():
    train_df, test_df = load_data()

    target_col = "median_house_value"

    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    X_raw = train_df.drop(columns=target_col)
    y_raw = train_df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    ridge_preprocess = make_pipeline(
        SimpleImputer(strategy="median"),
        PowerTransformer(method="yeo-johnson", standardize=True),
        StandardScaler(with_mean=True, with_std=True),
    )

    hgb_preprocess = SimpleImputer(strategy="median")

    hgb_model_shallow = make_pipeline(
        hgb_preprocess,
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=4,
            max_iter=350,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
    )

    hgb_model_deeper = make_pipeline(
        hgb_preprocess,
        HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=8,
            max_iter=700,
            min_samples_leaf=15,
            l2_regularization=0.0,
            random_state=43,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
    )

    ridge_model = make_pipeline(
        ridge_preprocess,
        Ridge(alpha=2.0, random_state=42),
    )

    hgb_model_shallow.fit(X_train, y_train)
    hgb_model_deeper.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)

    valid_pred_shallow = hgb_model_shallow.predict(X_valid)
    valid_pred_deeper = hgb_model_deeper.predict(X_valid)
    valid_pred_ridge = ridge_model.predict(X_valid)

    rmse_shallow = root_mean_squared_error(y_valid, valid_pred_shallow)
    rmse_deeper = root_mean_squared_error(y_valid, valid_pred_deeper)
    rmse_ridge = root_mean_squared_error(y_valid, valid_pred_ridge)

    eps = 1e-8
    inv_rmse_shallow = 1.0 / (rmse_shallow + eps)
    inv_rmse_deeper = 1.0 / (rmse_deeper + eps)
    inv_rmse_ridge = 1.0 / (rmse_ridge + eps)

    weight_sum = inv_rmse_shallow + inv_rmse_deeper + inv_rmse_ridge
    w_shallow = inv_rmse_shallow / weight_sum
    w_deeper = inv_rmse_deeper / weight_sum
    w_ridge = inv_rmse_ridge / weight_sum

    valid_pred_ensemble = (
        w_shallow * valid_pred_shallow
        + w_deeper * valid_pred_deeper
        + w_ridge * valid_pred_ridge
    )

    final_validation_score = root_mean_squared_error(y_valid, valid_pred_ensemble)
    print(f"Final Validation Performance: {final_validation_score}")

    _ = X.skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=8,
            max_iter=700,
            min_samples_leaf=15,
            l2_regularization=0.0,
            random_state=7,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
        y=y,
    )

    hgb_model_shallow.fit(X_raw, y_raw)
    hgb_model_deeper.fit(X_raw, y_raw)
    ridge_model.fit(X_raw, y_raw)

    test_pred_shallow = hgb_model_shallow.predict(test_df)
    test_pred_deeper = hgb_model_deeper.predict(test_df)
    test_pred_ridge = ridge_model.predict(test_df)

    test_pred = (
        w_shallow * test_pred_shallow
        + w_deeper * test_pred_deeper
        + w_ridge * test_pred_ridge
    )

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
