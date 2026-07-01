
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

    # Robustly handle contexts where target may or may not be present
    feature_cols = [c for c in train_df.columns if c != target_col]

    # Keep DataOps structure intact
    data = skrub.var("data", train_df)
    X = data[feature_cols].skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    regressor = HistGradientBoostingRegressor(random_state=0)

    pred = X.skb.apply(vectorizer).skb.apply(regressor, y=y)

    # Holdout validation for reporting
    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=0)
    train_env = {"data": train_part}
    valid_env = {"data": valid_part}

    learner = pred.skb.make_learner(fitted=True)
    learner.fit(train_env)

    valid_pred = learner.predict(valid_env)
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    return learner


def main():
    train_df, test_df = load_data()
    learner = build_and_evaluate(train_df)

    test_pred = learner.predict({"data": test_df})
    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
