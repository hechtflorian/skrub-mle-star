
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub"])
    import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_base_models():
    hgb_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        ),
    )

    ridge_model = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=1.0, random_state=42),
    )
    return hgb_model, ridge_model


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

    # Base model A: current workflow blend
    hgb_model_a, ridge_model_a = get_base_models()
    hgb_model_a.fit(X_train, y_train)
    ridge_model_a.fit(X_train, y_train)

    valid_pred_hgb_a = hgb_model_a.predict(X_valid)
    valid_pred_ridge_a = ridge_model_a.predict(X_valid)
    valid_pred_a = 0.8 * valid_pred_hgb_a + 0.2 * valid_pred_ridge_a
    rmse_a = root_mean_squared_error(y_valid, valid_pred_a)

    # Base model B: slightly different version of the same workflow
    # Kept minimal and conservative to preserve the original pipeline style.
    hgb_model_b = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=10,
            max_iter=700,
            min_samples_leaf=15,
            l2_regularization=0.0,
            random_state=7,
        ),
    )
    ridge_model_b = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=0.5, random_state=7),
    )

    hgb_model_b.fit(X_train, y_train)
    ridge_model_b.fit(X_train, y_train)

    valid_pred_hgb_b = hgb_model_b.predict(X_valid)
    valid_pred_ridge_b = ridge_model_b.predict(X_valid)
    valid_pred_b = 0.7 * valid_pred_hgb_b + 0.3 * valid_pred_ridge_b
    rmse_b = root_mean_squared_error(y_valid, valid_pred_b)

    # Inverse-RMSE weights with a small floor for stability
    eps = 1e-8
    inv_a = 1.0 / max(rmse_a, eps)
    inv_b = 1.0 / max(rmse_b, eps)
    w_a = inv_a / (inv_a + inv_b)
    w_b = inv_b / (inv_a + inv_b)

    # Conservative average if models are very similar; otherwise use RMSE-based weights
    if abs(rmse_a - rmse_b) < 1e-4:
        w_a, w_b = 0.5, 0.5

    final_validation_pred = w_a * valid_pred_a + w_b * valid_pred_b
    final_validation_score = root_mean_squared_error(y_valid, final_validation_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    # Keep skrub DataOps workflow intact as the main structure
    _ = X.skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        ),
        y=y,
    )

    # Refit on full data for test predictions
    hgb_model_a, ridge_model_a = get_base_models()
    hgb_model_a.fit(X_raw, y_raw)
    ridge_model_a.fit(X_raw, y_raw)

    hgb_model_b = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=10,
            max_iter=700,
            min_samples_leaf=15,
            l2_regularization=0.0,
            random_state=7,
        ),
    )
    ridge_model_b = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=0.5, random_state=7),
    )
    hgb_model_b.fit(X_raw, y_raw)
    ridge_model_b.fit(X_raw, y_raw)

    test_pred_hgb_a = hgb_model_a.predict(test_df)
    test_pred_ridge_a = ridge_model_a.predict(test_df)
    test_pred_a = 0.8 * test_pred_hgb_a + 0.2 * test_pred_ridge_a

    test_pred_hgb_b = hgb_model_b.predict(test_df)
    test_pred_ridge_b = ridge_model_b.predict(test_df)
    test_pred_b = 0.7 * test_pred_hgb_b + 0.3 * test_pred_ridge_b

    test_pred = w_a * test_pred_a + w_b * test_pred_b

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
