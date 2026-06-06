
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline(train_df, test_df):
    target_col = "median_house_value"

    train_split, valid_split = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    data = skrub.var("data", train_split)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1000,
        depth=8,
        learning_rate=0.03,
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
    )

    learner_plan = X.skb.apply(vectorizer).skb.apply(model, y=y)
    learner = learner_plan.skb.make_learner(fitted=True)

    X_valid = valid_split.drop(columns=[target_col], errors="ignore")
    y_valid = valid_split[target_col].values
    valid_preds = learner.predict({"data": X_valid})
    final_validation_score = mean_squared_error(y_valid, valid_preds) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    full_data = skrub.var("data", train_df)
    X_full = full_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_full = full_data[target_col].skb.mark_as_y()
    full_plan = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
    full_learner = full_plan.skb.make_learner(fitted=True)

    test_preds = full_learner.predict({"data": test_df})
    return test_preds


def main():
    train_df, test_df = load_data()
    preds = build_pipeline(train_df, test_df)
    submission = pd.DataFrame({"median_house_value": preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
