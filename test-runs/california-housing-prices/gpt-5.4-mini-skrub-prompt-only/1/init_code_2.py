
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import skrub as skb


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Use skrub DataOps correctly from pandas inputs
    df = skb.var(train_df)

    # Mark X / y with skrub APIs without relying on a pandas .skb accessor
    X = skb.X(df.drop(columns=["median_house_value"]))
    y = skb.y(df["median_house_value"])

    # Materialize to pandas for model training
    X_raw = train_df.drop(columns=["median_house_value"])
    y_raw = train_df["median_house_value"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    numeric_features = X_train.columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    valid_pred = pipe.predict(X_valid)
    final_validation_score = rmse(y_valid, valid_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    pipe.fit(X_raw, y_raw)
    test_pred = pipe.predict(test_df)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)

    _ = (X, y)


if __name__ == "__main__":
    main()
