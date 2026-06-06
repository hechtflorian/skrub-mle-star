
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_and_eval(train_df):
    target_col = "median_house_value"

    # Keep a small subsample only for fast iteration if desired, but do not remove it.
    data = skrub.var("data", train_df)
    if len(train_df) > 5000:
        data = data.skb.subsample(n=5000)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Fix: use a proper DataOps/skrub call, not pandas .skb access on a raw DataFrame.
    # Cleaner is applied directly in the DataOps graph.
    X_clean = X.skb.apply(skrub.Cleaner(drop_if_constant=True))

    # Robust baseline model for tabular regression.
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pred = X_clean.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y)

    # Holdout validation on the available training data.
    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)
    train_data = skrub.var("data", train_part)
    valid_data = skrub.var("data", valid_part)

    X_train = train_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_data[target_col].skb.mark_as_y()
    X_train = X_train.skb.apply(skrub.Cleaner(drop_if_constant=True))
    learner = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train).skb.make_learner(fitted=True)

    X_valid = valid_data.drop(columns=target_col, errors="ignore")
    y_valid = valid_part[target_col].values
    y_pred = learner.predict({"data": valid_part.drop(columns=[target_col], errors="ignore")})
    score = rmse(y_valid, y_pred)

    print(f"Final Validation Performance: {score}")
    return learner


def predict_and_save(learner, test_df):
    preds = learner.predict({"data": test_df})
    submission = pd.DataFrame({"median_house_value": preds})
    submission.to_csv("submission.csv", index=False)
    return submission


def main():
    train_df, test_df = load_data()
    learner = build_and_eval(train_df)
    predict_and_save(learner, test_df)


if __name__ == "__main__":
    main()
