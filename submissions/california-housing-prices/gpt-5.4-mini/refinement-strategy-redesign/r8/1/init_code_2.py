
import os
import numpy as np
import pandas as pd
import skrub

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    train_data = skrub.var("data", train_df)
    X = train_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = train_data[target_col].skb.mark_as_y()

    X_pipeline = X.skb.apply(skrub.TableVectorizer())

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )

    pred_graph = X_pipeline.skb.apply(model, y=y)

    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)
    train_part_data = skrub.var("data", train_part)
    valid_part_data = skrub.var("data", valid_part)

    X_train = train_part_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_part_data[target_col].skb.mark_as_y()
    X_train_pipeline = X_train.skb.apply(skrub.TableVectorizer())
    train_pred_graph = X_train_pipeline.skb.apply(model, y=y_train)

    learner = train_pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    full_learner = pred_graph.skb.make_learner(fitted=True)
    test_pred = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
