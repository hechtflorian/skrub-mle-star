
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import skrub


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Keep the DataOps structure, but fix the invalid `.skb.apply(lambda ...)`
    # by using `.skb.apply_func(...)` for plain Python functions.
    X_pipeline = X.skb.apply_func(lambda df: df)

    # Reuse a robust sklearn model while preserving the DataOps pipeline shape.
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pred = X_pipeline.skb.apply(model, y=y)

    # Validation split for reporting
    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)
    train_part_data = skrub.var("data", train_part)
    valid_part_data = skrub.var("data", valid_part)

    X_train = train_part_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_part_data[target_col].skb.mark_as_y()

    X_train_pipeline = X_train.skb.apply_func(lambda df: df)
    pred_train = X_train_pipeline.skb.apply(model, y=y_train)

    learner = pred_train.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full training data and predict test
    full_learner = pred.skb.make_learner(fitted=True)
    test_pred = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head())


if __name__ == "__main__":
    main()
