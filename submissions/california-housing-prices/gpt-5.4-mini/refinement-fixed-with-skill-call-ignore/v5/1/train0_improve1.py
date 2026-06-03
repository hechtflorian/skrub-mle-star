
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

    train_df = train_df.copy()

    # Fill missing values in a simple, robust way before DataOps graph construction.
    feature_cols = [c for c in train_df.columns if c != target_col]
    for col in feature_cols:
        if train_df[col].isna().any():
            if pd.api.types.is_numeric_dtype(train_df[col]):
                train_df[col] = train_df[col].fillna(train_df[col].median())
            else:
                train_df[col] = train_df[col].fillna(train_df[col].mode().iloc[0])

    if train_df[target_col].isna().any():
        train_df = train_df.dropna(subset=[target_col])

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    data = skrub.var("data", train_split)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    regressor = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
    )
    pred = X_vec.skb.apply(regressor, y=y)

    learner = pred.skb.make_learner(fitted=True)
    val_pred = learner.predict({"data": val_split})

    final_validation_score = mean_squared_error(
        val_split[target_col].to_numpy(), np.asarray(val_pred)
    ) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")
    return learner, final_validation_score


def predict_test(learner, test_df):
    test_df = test_df.copy()
    for col in test_df.columns:
        if test_df[col].isna().any():
            if pd.api.types.is_numeric_dtype(test_df[col]):
                test_df[col] = test_df[col].fillna(test_df[col].median())
            else:
                test_df[col] = test_df[col].fillna(test_df[col].mode().iloc[0])

    test_pred = learner.predict({"data": test_df})
    return np.asarray(test_pred)


def main():
    train_df, test_df = load_data()
    learner, _ = build_and_evaluate(train_df)
    test_pred = predict_test(learner, test_df)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
