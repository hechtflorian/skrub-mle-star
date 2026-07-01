
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Keep the original DataOps pipeline shape intact.
    X_pipeline = X.skb.apply_func(lambda df: df)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pred = X_pipeline.skb.apply(model, y=y)

    # Validation split for reporting and prediction-space blending.
    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

    train_part_data = skrub.var("data", train_part)
    valid_part_data = skrub.var("data", valid_part)

    X_train = train_part_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_part_data[target_col].skb.mark_as_y()

    X_train_pipeline = X_train.skb.apply_func(lambda df: df)
    pred_train = X_train_pipeline.skb.apply(model, y=y_train)

    learner = pred_train.skb.make_learner(fitted=True)

    valid_X_env = {"data": valid_part.drop(columns=target_col, errors="ignore")}
    valid_pred_base = learner.predict(valid_X_env)

    y_valid = valid_part[target_col].values
    y_train_values = train_part[target_col].values

    q01_train = np.quantile(y_train_values, 0.01)
    q05_train = np.quantile(y_train_values, 0.05)
    q10_train = np.quantile(y_train_values, 0.10)
    q95_train = np.quantile(y_train_values, 0.95)
    mean_train_target = float(np.mean(y_train_values))

    # Base / clipped / shrunk predictions on validation.
    p_base = np.asarray(valid_pred_base)
    p_clip_01 = np.clip(p_base, q01_train, q95_train)
    p_clip_05 = np.clip(p_base, q05_train, q95_train)
    p_clip_10 = np.clip(p_base, q10_train, q95_train)
    p_clip = (p_clip_01 + p_clip_05 + p_clip_10) / 3.0
    p_shrink = 0.9 * p_base + 0.1 * mean_train_target

    rmse_base = rmse(y_valid, p_base)
    rmse_clip = rmse(y_valid, p_clip)
    rmse_shrink = rmse(y_valid, p_shrink)

    scores = {
        "base": rmse_base,
        "clip": rmse_clip,
        "shrink": rmse_shrink,
    }
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    best_name, best_score = sorted_items[0]
    second_name, second_score = sorted_items[1]

    # Tiny weighted average if top two are very close; otherwise choose the best transform.
    close_threshold = 0.005 * best_score
    if (second_score - best_score) <= close_threshold:
        inv1 = 1.0 / (sorted_items[0][1] + 1e-12)
        inv2 = 1.0 / (sorted_items[1][1] + 1e-12)
        w1 = inv1 / (inv1 + inv2)
        w2 = inv2 / (inv1 + inv2)
        chosen_transforms = [sorted_items[0][0], sorted_items[1][0]]
        chosen_weights = [w1, w2]
    else:
        chosen_transforms = [best_name]
        chosen_weights = [1.0]

    # Fit on full training data and predict test.
    full_learner = pred.skb.make_learner(fitted=True)

    test_X_env = {"data": test_df}
    test_pred_base = np.asarray(full_learner.predict(test_X_env))
    test_p_clip_01 = np.clip(test_pred_base, q01_train, q95_train)
    test_p_clip_05 = np.clip(test_pred_base, q05_train, q95_train)
    test_p_clip_10 = np.clip(test_pred_base, q10_train, q95_train)
    test_p_clip = (test_p_clip_01 + test_p_clip_05 + test_p_clip_10) / 3.0
    test_p_shrink = 0.9 * test_pred_base + 0.1 * mean_train_target

    transform_map = {
        "base": test_pred_base,
        "clip": test_p_clip,
        "shrink": test_p_shrink,
    }

    final_test_pred = np.zeros_like(test_pred_base, dtype=float)
    for t_name, w in zip(chosen_transforms, chosen_weights):
        final_test_pred += w * transform_map[t_name]

    valid_final_pred = np.zeros_like(p_base, dtype=float)
    valid_transform_map = {
        "base": p_base,
        "clip": p_clip,
        "shrink": p_shrink,
    }
    for t_name, w in zip(chosen_transforms, chosen_weights):
        valid_final_pred += w * valid_transform_map[t_name]

    final_validation_score = rmse(y_valid, valid_final_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({"median_house_value": final_test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head())


if __name__ == "__main__":
    main()
