
import os
import numpy as np
import pandas as pd

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TARGET_COL = "median_house_value"
TRAIN_PATH = os.path.join("./input", "train.csv")
TEST_PATH = os.path.join("./input", "test.csv")
SUBMISSION_PATH = "submission.csv"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_dataops_model(data, target_col=TARGET_COL):
    # Fix: only drop target if it actually exists in the provided dataframe.
    # This preserves the DataOps-first workflow while avoiding KeyError on test-like inputs.
    if target_col in data.columns:
        X_df = data.drop(columns=[target_col])
        y = data[target_col].skb.mark_as_y()
    else:
        X_df = data
        y = None

    X = X_df.skb.mark_as_X()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    reg = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )

    pred = X_vec.skb.apply(reg, y=y) if y is not None else X_vec.skb.apply(reg)
    return pred


def fit_and_evaluate(train_df):
    train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=42)

    pred_graph = build_dataops_model(train_part, target_col=TARGET_COL)
    learner = pred_graph.skb.make_learner(fitted=False)
    learner.fit({"data": train_part})

    val_features = val_part.drop(columns=[TARGET_COL])
    val_pred = learner.predict({"data": val_features})

    final_validation_score = mean_squared_error(
        val_part[TARGET_COL].values, np.asarray(val_pred)
    ) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")
    return learner, final_validation_score


def train_full_and_predict(train_df, test_df):
    pred_graph = build_dataops_model(train_df, target_col=TARGET_COL)
    learner = pred_graph.skb.make_learner(fitted=False)
    learner.fit({"data": train_df})

    test_pred = learner.predict({"data": test_df})
    return np.asarray(test_pred)


def main():
    train_df, test_df = load_data()

    _, _ = fit_and_evaluate(train_df)
    test_predictions = train_full_and_predict(train_df, test_df)

    submission = pd.DataFrame({TARGET_COL: test_predictions})
    submission.to_csv(SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
