
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    # Keep a stable split for validation without rewriting the DataOps workflow.
    rng = np.random.RandomState(42)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx = idx[:split]
    valid_idx = idx[split:]

    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    valid_split = train_df.iloc[valid_idx].reset_index(drop=True)

    # DataOps-first graph
    data = skrub.var("data", train_split)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    regressor = HistGradientBoostingRegressor(random_state=42)
    pred = X_vec.skb.apply(regressor, y=y)

    learner = pred.skb.make_learner(fitted=True)

    # Fix: do NOT drop target from valid_split during prediction; it already doesn't contain it.
    # Pass the environment dict with the raw feature dataframe.
    valid_features = valid_split.drop(columns=[target_col], errors="ignore")
    valid_preds = learner.predict({"data": valid_features})

    final_validation_score = rmse(valid_split[target_col].to_numpy(), np.asarray(valid_preds))
    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full training data and predict test for submission
    full_data = skrub.var("data", train_df)
    full_X = full_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    full_y = full_data[target_col].skb.mark_as_y()

    full_pred = full_X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=42), y=full_y
    )
    full_learner = full_pred.skb.make_learner(fitted=True)

    test_preds = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": np.asarray(test_preds)})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
