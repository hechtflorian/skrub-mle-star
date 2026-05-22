import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


def load_data():
    train_path = os.path.join("./input", "train.csv")
    train_df = pd.read_csv(train_path)
    return train_df


def train_and_eval(model, X_train, y_train, X_valid, y_valid, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, preds)
    print(f"{name} RMSE: {rmse:.6f}")
    return rmse


def main():
    train_df = load_data()

    target_col = "median_house_value"
    X = train_df.drop(columns=target_col)
    y = train_df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_hgb = make_pipeline(
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

    no_impute_hgb = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    ridge_model = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=1.0, random_state=42),
    )

    print("Ablation study results:")
    base_rmse = train_and_eval(base_hgb, X_train, y_train, X_valid, y_valid, "Base HGB + median imputation")

    ablation_1_rmse = train_and_eval(
        no_impute_hgb, X_train, y_train, X_valid, y_valid, "Ablation 1: HGB without imputation"
    )

    ablation_2_rmse = train_and_eval(
        ridge_model, X_train, y_train, X_valid, y_valid, "Ablation 2: Ridge + median imputation"
    )

    delta_no_impute = ablation_1_rmse - base_rmse
    delta_ridge = ablation_2_rmse - base_rmse

    print("\nImpact relative to base model:")
    print(f"Removing imputation changes RMSE by: {delta_no_impute:+.6f}")
    print(f"Replacing HGB with Ridge changes RMSE by: {delta_ridge:+.6f}")

    impacts = {
        "median imputation": delta_no_impute,
        "HistGradientBoostingRegressor vs Ridge": delta_ridge,
    }

    most_important = max(impacts, key=lambda k: abs(impacts[k]))
    print(f"\nMost important part contributing to performance: {most_important}")


if __name__ == "__main__":
    main()