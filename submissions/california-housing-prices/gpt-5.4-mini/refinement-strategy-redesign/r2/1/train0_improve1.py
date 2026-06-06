
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

    # Fast preview subsampling is allowed, but keep the final workflow on full data.
    data = skrub.var("data", train_df)
    data_preview = data.skb.subsample(n=min(5000, len(train_df)))

    X = data_preview.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_preview[target_col].skb.mark_as_y()

    # Use a robust tree-based regressor that can handle the default TableVectorizer output well.
    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=None,
        max_iter=250,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)

    # Evaluate on the previewed subsample for quick validation during development.
    cv = pred.skb.cross_validate(keep_subsampling=True)
    preview_rmse = float(np.mean(cv["test_score"]) ** 0.5) if "test_score" in cv else float("nan")

    # Final model on full data
    full_data = skrub.var("data", train_df)
    full_X = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    full_y = full_data[target_col].skb.mark_as_y()

    full_pred = full_X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=None,
            max_iter=250,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        ),
        y=full_y,
    )

    learner = full_pred.skb.make_learner(fitted=True)
    return learner, preview_rmse


def main():
    train_df, test_df = load_data()
    learner, preview_rmse = build_and_evaluate(train_df)

    # Fit/predict through the DataOps learner with an environment dict.
    preds = learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": np.asarray(preds).ravel()})
    submission.to_csv("submission.csv", index=False)

    final_validation_score = preview_rmse
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
