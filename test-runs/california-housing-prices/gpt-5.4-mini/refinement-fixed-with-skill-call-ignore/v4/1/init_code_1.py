
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_and_validate(train_df):
    target_col = "median_house_value"

    # Keep DataOps-first structure, but make the environment key explicit.
    data = skrub.var("data", train_df)

    X_train_df = train_df.drop(columns=[target_col])
    y_train_df = train_df[target_col]

    X = skrub.X(X_train_df)
    y = skrub.y(y_train_df)

    # Simple, robust preprocessing/model in DataOps graph.
    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        random_state=42,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)

    # Validation split
    X_tr_df, X_val_df, y_tr_df, y_val_df = train_test_split(
        X_train_df, y_train_df, test_size=0.2, random_state=42
    )

    # IMPORTANT: provide the key expected by skrub.X(...), i.e. "X"
    train_env = {"X": X_tr_df, "y": y_tr_df}
    val_env = {"X": X_val_df, "y": y_val_df}

    learner = pred.skb.make_learner()
    learner.fit(train_env)

    val_pred = learner.predict(val_env)
    final_validation_score = mean_squared_error(y_val_df, val_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    return learner


def train_and_predict(train_df, test_df):
    target_col = "median_house_value"

    X_train_df = train_df.drop(columns=[target_col])
    y_train_df = train_df[target_col]

    X = skrub.X(X_train_df)
    y = skrub.y(y_train_df)

    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        random_state=42,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
    learner = pred.skb.make_learner(fitted=True)

    # Fit on full training data using the correct environment keys
    learner.fit({"X": X_train_df, "y": y_train_df})

    test_pred = learner.predict({"X": test_df})
    return test_pred


def main():
    train_df, test_df = load_data()
    _ = build_and_validate(train_df)
    test_pred = train_and_predict(train_df, test_df)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
