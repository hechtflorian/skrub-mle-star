
import os
import warnings

import numpy as np
import pandas as pd
import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_and_evaluate(train_df):
    target_col = "median_house_value"

    # Fast iteration subsampling for preview/debugging, preserved as requested.
    data = skrub.var("data", train_df).skb.subsample(n=min(5000, len(train_df)))

    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    pred = X.skb.apply(vectorizer).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=200,
            min_samples_leaf=20,
            random_state=0,
        ),
        y=y,
    )

    # Fit on the full training data through the DataOps learner interface.
    learner = pred.skb.make_learner(fitted=False)
    learner.fit({"data": train_df})

    return learner


def main():
    train_df, test_df = load_data()
    target_col = "median_house_value"

    # Train/validation split for an actual validation score.
    train_part, valid_part = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    learner = build_and_evaluate(train_part)

    valid_pred = learner.predict({"data": valid_part})
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # Refit on full training data before test prediction.
    full_learner = build_and_evaluate(train_df)

    test_pred = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
