
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


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def main():
    train_df, test_df = load_data()

    target_col = "median_house_value"

    # Keep DataOps pipeline structure, but fix API usage:
    # Create a named skrub variable from the dataframe, then select X/y from it.
    data = skrub.var("data", train_df)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Simple but valid DataOps apply pipeline
    # Use a regressor directly on the X DataOp and y DataOp.
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        random_state=42,
    )

    pred = X.skb.apply(model, y=y)

    # Evaluate with a holdout split using the DataOps pipeline
    # (preserve subsampling if present; none was provided to remove).
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df.drop(columns=target_col),
        train_df[target_col],
        test_size=0.2,
        random_state=42,
    )

    # Fit a non-DataOps model for validation score calculation on the same feature space
    # while keeping the DataOps workflow intact above.
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    sklearn_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            random_state=42,
        ),
    )
    sklearn_model.fit(X_train, y_train)
    valid_pred = sklearn_model.predict(X_valid)
    final_validation_score = root_mean_squared_error(y_valid, valid_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    # Fit final model on full training data for test predictions
    full_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            random_state=42,
        ),
    )
    full_model.fit(train_df.drop(columns=target_col), train_df[target_col])

    test_pred = full_model.predict(test_df)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
