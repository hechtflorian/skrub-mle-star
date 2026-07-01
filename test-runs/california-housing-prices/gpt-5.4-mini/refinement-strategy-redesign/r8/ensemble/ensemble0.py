
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub


def fit_and_predict(train_df, test_df, target_col, split_seed=42, model_seed=42):
    # Base pipeline kept intact
    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    X_pipeline = X.skb.apply_func(lambda df: df)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=model_seed,
    )

    pred = X_pipeline.skb.apply(model, y=y)

    # Validation split for this variant
    train_part, valid_part = train_test_split(
        train_df, test_size=0.2, random_state=split_seed
    )

    train_part_data = skrub.var("data", train_part)
    valid_part_data = skrub.var("data", valid_part)

    X_train = train_part_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_part_data[target_col].skb.mark_as_y()

    X_train_pipeline = X_train.skb.apply_func(lambda df: df)
    pred_train = X_train_pipeline.skb.apply(model, y=y_train)

    learner = pred_train.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5

    # Fit on full training data and predict test
    full_learner = pred.skb.make_learner(fitted=True)
    test_pred = full_learner.predict({"data": test_df})

    return test_pred, rmse


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    # Internal ensemble with safe variations
    variants = [
        {"split_seed": 42, "model_seed": 42},
        {"split_seed": 7, "model_seed": 7},
    ]

    test_preds = []
    rmses = []

    for i, params in enumerate(variants, start=1):
        test_pred_i, rmse_i = fit_and_predict(
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            split_seed=params["split_seed"],
            model_seed=params["model_seed"],
        )
        test_preds.append(np.asarray(test_pred_i))
        rmses.append(rmse_i)
        print(f"Ablation[variant_{i}] RMSE: {rmse_i}")

    # Weighted average if validation is used; weights proportional to inverse RMSE
    rmses = np.asarray(rmses, dtype=float)
    weights = 1.0 / np.maximum(rmses, 1e-12)
    weights = weights / weights.sum()

    final_test_pred = np.average(np.vstack(test_preds), axis=0, weights=weights)

    final_validation_score = float(np.average(rmses, weights=weights))
    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({"median_house_value": final_test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head())


if __name__ == "__main__":
    main()
