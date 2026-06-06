
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import skrub
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    train_data = skrub.var("data", train_df)
    X = train_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = train_data[target_col].skb.mark_as_y()

    # Main DataOps preprocessing: robust tabular vectorization
    vectorizer = skrub.TableVectorizer()
    X_pipeline = X.skb.apply(vectorizer)

    # Two complementary tree models to ensemble
    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )

    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pred_lgbm = X_pipeline.skb.apply(lgbm, y=y)
    pred_hgb = X_pipeline.skb.apply(hgb, y=y)

    # Hold-out validation
    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)
    train_part_data = skrub.var("data", train_part)
    valid_part_data = skrub.var("data", valid_part)

    X_train = train_part_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_part_data[target_col].skb.mark_as_y()

    X_train_pipeline = X_train.skb.apply(skrub.TableVectorizer())
    train_pred_lgbm = X_train_pipeline.skb.apply(lgbm, y=y_train)
    train_pred_hgb = X_train_pipeline.skb.apply(hgb, y=y_train)

    learner_lgbm = train_pred_lgbm.skb.make_learner(fitted=True)
    learner_hgb = train_pred_hgb.skb.make_learner(fitted=True)

    valid_pred_lgbm = learner_lgbm.predict({"data": valid_part})
    valid_pred_hgb = learner_hgb.predict({"data": valid_part})

    # Simple ensemble of the two model predictions
    valid_pred = 0.5 * valid_pred_lgbm + 0.5 * valid_pred_hgb
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full training data and predict test
    full_learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
    full_learner_hgb = pred_hgb.skb.make_learner(fitted=True)

    test_pred_lgbm = full_learner_lgbm.predict({"data": test_df})
    test_pred_hgb = full_learner_hgb.predict({"data": test_df})

    test_pred = 0.5 * test_pred_lgbm + 0.5 * test_pred_hgb

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head())


if __name__ == "__main__":
    main()
