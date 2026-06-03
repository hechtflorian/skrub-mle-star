
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET = "median_house_value"


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_and_evaluate(train_df):
    data = train_df.copy()

    # Keep subsampling if present, but do not rely on it for fitting the final learner.
    if len(data) > 5000:
        preview_data = data.sample(n=5000, random_state=RANDOM_STATE)
    else:
        preview_data = data

    # DataOps graph
    data_var = skrub.var("data", preview_data)
    X = data_var.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data_var[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred = X_vec.skb.apply(regressor, y=y)

    # Build learner from DataOps graph
    learner = pred.skb.make_learner(fitted=False)

    # Fit on the same environment used to define X and y
    learner.fit({"data": preview_data})

    # Validation split for evaluation
    train_split, val_split = train_test_split(
        data, test_size=0.2, random_state=RANDOM_STATE
    )
    val_preds = learner.predict({"data": val_split})
    final_validation_score = rmse(val_split[TARGET], val_preds)

    print(f"Final Validation Performance: {final_validation_score}")
    return learner, final_validation_score


def fit_full_model(train_df):
    data = train_df.copy()

    data_var = skrub.var("data", data)
    X = data_var.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data_var[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred = X_vec.skb.apply(regressor, y=y)
    learner = pred.skb.make_learner(fitted=False)
    learner.fit({"data": data})
    return learner


def main():
    train_df, test_df = load_data()

    _, final_validation_score = build_and_evaluate(train_df)

    final_learner = fit_full_model(train_df)
    test_preds = final_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_preds})

    os.makedirs("./final", exist_ok=True)
    submission.to_csv("./final/submission.csv", index=False)

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
