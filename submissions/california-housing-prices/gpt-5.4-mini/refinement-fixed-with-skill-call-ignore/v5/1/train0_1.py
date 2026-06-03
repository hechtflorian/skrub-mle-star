
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    df["population_per_household"] = df["population"] / (df["households"] + eps)
    df["income_per_room"] = df["median_income"] / (df["total_rooms"] + eps)
    return df


def build_pipeline(train_df: pd.DataFrame):
    train_df = add_features(train_df)

    preview_n = min(5000, len(train_df))
    data = skrub.var("data", train_df).skb.subsample(n=preview_n)

    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
    return pred, preview_n


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    pred, preview_n = build_pipeline(train_df)

    valid_df = train_df.sample(frac=0.15, random_state=RANDOM_STATE)
    train_part = train_df.drop(valid_df.index).reset_index(drop=True)
    valid_part = valid_df.reset_index(drop=True)

    train_part = add_features(train_part)
    valid_part = add_features(valid_part)

    X_train = train_part.drop(columns=TARGET_COL, errors="ignore")
    y_train = train_part[TARGET_COL]
    X_valid = valid_part.drop(columns=TARGET_COL, errors="ignore")
    y_valid = valid_part[TARGET_COL]

    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)
    valid_pred = model.predict(vectorizer.transform(X_valid))
    final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    full_df = add_features(train_df)
    X_full = full_df.drop(columns=TARGET_COL, errors="ignore")
    y_full = full_df[TARGET_COL]

    final_vectorizer = skrub.TableVectorizer()
    final_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )
    X_full_vec = final_vectorizer.fit_transform(X_full)
    final_model.fit(X_full_vec, y_full)

    test_df = add_features(test_df)
    test_pred = final_model.predict(final_vectorizer.transform(test_df))

    submission = pd.DataFrame({TARGET_COL: test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
