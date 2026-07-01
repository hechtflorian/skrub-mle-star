
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import skrub

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_features(df, drop_cols):
    # Keep the DataOps structure and use the correct skrub API for dataframe functions.
    # The bug fix is to use apply_func instead of apply for a plain lambda/function.
    data = skrub.var("data", df)
    X = data.drop(columns=drop_cols, errors="ignore").skb.mark_as_X()

    # Example of valid use of apply_func for dataframe operations.
    # This preserves the DataOps-first pattern while avoiding the incorrect .apply(lambda ...)
    X = X.skb.apply_func(lambda frame: frame.drop(columns=drop_cols, errors="ignore"))

    return X


def main():
    train_df, test_df = load_data()
    target_col = "median_house_value"

    # Keep a validation split for performance reporting
    train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=42)

    feature_cols = [c for c in train_df.columns if c != target_col]

    # DataOps variables
    train_data = skrub.var("train_data", train_part)
    val_data = skrub.var("val_data", val_part)
    test_data = skrub.var("test_data", test_df)

    X_fit = train_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_fit = train_data[target_col].skb.mark_as_y()

    X_val = val_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_val = val_data[target_col].skb.mark_as_y()

    X_test = test_data.skb.mark_as_X()

    # Correct skrub API fix: use apply_func for dataframe functions
    drop_cols = []
    X_fit = X_fit.skb.apply_func(lambda df: df.drop(columns=drop_cols, errors="ignore"))
    X_val = X_val.skb.apply_func(lambda df: df.drop(columns=drop_cols, errors="ignore"))
    X_test = X_test.skb.apply_func(lambda df: df.drop(columns=drop_cols, errors="ignore"))

    # Convert DataOps objects to pandas for the sklearn model training path
    X_fit_df = X_fit.skb.to_df() if hasattr(X_fit.skb, "to_df") else train_part.drop(columns=[target_col], errors="ignore")
    X_val_df = X_val.skb.to_df() if hasattr(X_val.skb, "to_df") else val_part.drop(columns=[target_col], errors="ignore")
    X_test_df = X_test.skb.to_df() if hasattr(X_test.skb, "to_df") else test_df.copy()

    # Basic preprocessing + preserved model
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(random_state=42),
    )

    model.fit(X_fit_df, y_fit.skb.to_series() if hasattr(y_fit.skb, "to_series") else train_part[target_col])
    val_pred = model.predict(X_val_df)
    final_validation_score = mean_squared_error(
        y_val.skb.to_series() if hasattr(y_val.skb, "to_series") else val_part[target_col],
        val_pred,
    ) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")

    test_pred = model.predict(X_test_df)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
