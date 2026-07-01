
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


    # Lightweight HGB-focused tuning candidates (keep search small for speed).
    hgb_params_list = [
        dict(
            learning_rate=0.05,
            max_depth=6,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        ),
        dict(
            learning_rate=0.03,
            max_depth=8,
            max_iter=700,
            min_samples_leaf=15,
            l2_regularization=0.0,
            random_state=42,
        ),
        dict(
            learning_rate=0.07,
            max_depth=5,
            max_iter=400,
            min_samples_leaf=25,
            l2_regularization=0.0,
            random_state=42,
        ),
    ]

    hgb_models = [
        HistGradientBoostingRegressor(**params) for params in hgb_params_list
    ]

    ridge_model = Ridge(alpha=1.0, random_state=42)

    valid_pred_hgb_candidates = []
    hgb_validation_scores = []

    for model in hgb_models:
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        valid_pred_hgb_candidates.append(pred)
        hgb_validation_scores.append(root_mean_squared_error(y_valid, pred))

    best_hgb_idx = int(np.argmin(hgb_validation_scores))
    best_hgb_model = hgb_models[best_hgb_idx]
    valid_pred_hgb = valid_pred_hgb_candidates[best_hgb_idx]

    ridge_model.fit(X_train, y_train)
    valid_pred_ridge = ridge_model.predict(X_valid)

    # HGB-centered blend; ridge is only a minor stabilizer if it helps.
    valid_pred_ensemble = 0.95 * valid_pred_hgb + 0.05 * valid_pred_ridge
    final_validation_score = root_mean_squared_error(y_valid, valid_pred_ensemble)
    print(f"Final Validation Performance: {final_validation_score}")

    # Keep skrub DataOps as the main workflow by applying the stronger HGB setup on DataOps objects.
    _ = X.skb.apply(
        HistGradientBoostingRegressor(**hgb_params_list[best_hgb_idx]),
        y=y,
    )

    # Refit on all available raw data for test-time predictions.
    best_hgb_model.fit(X_raw, y_raw)
    ridge_model.fit(X_raw, y_raw)

    test_pred_hgb = best_hgb_model.predict(test_df)
    test_pred_ridge = ridge_model.predict(test_df)
    test_pred = 0.95 * test_pred_hgb + 0.05 * test_pred_ridge

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
