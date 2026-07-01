import os
import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def load_data():
    return pd.read_csv(TRAIN_PATH)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def make_dataset(train_df, subsample_n=5000):
    data = skrub.var("data", train_df).skb.subsample(n=min(subsample_n, len(train_df)))
    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()
    return X, y


def fit_predict_with_config(train_split, valid_split, use_vectorizer=True, use_subsample=True, max_iter=250):
    data = skrub.var("data", train_split)
    if use_subsample:
        data = data.skb.subsample(n=min(5000, len(train_split)))

    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    if use_vectorizer:
        X = X.skb.apply(skrub.TableVectorizer())

    pred = X.skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            max_iter=max_iter,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y,
    )
    learner = pred.skb.make_learner(fitted=True)
    valid_preds = learner.predict({"data": valid_split})
    return rmse(valid_split[TARGET_COL], valid_preds)


def main():
    train_df = load_data()

    idx = np.arange(len(train_df))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    train_split = train_df.iloc[tr_idx].reset_index(drop=True)
    valid_split = train_df.iloc[va_idx].reset_index(drop=True)

    results = {}

    # Baseline: original core recipe
    results["baseline"] = fit_predict_with_config(
        train_split, valid_split, use_vectorizer=True, use_subsample=True, max_iter=250
    )

    # Ablation 1: remove subsampling
    results["no_subsample"] = fit_predict_with_config(
        train_split, valid_split, use_vectorizer=True, use_subsample=False, max_iter=250
    )

    # Ablation 2: remove TableVectorizer
    results["no_vectorizer"] = fit_predict_with_config(
        train_split, valid_split, use_vectorizer=False, use_subsample=True, max_iter=250
    )

    # Ablation 3: smaller model
    results["fewer_trees"] = fit_predict_with_config(
        train_split, valid_split, use_vectorizer=True, use_subsample=True, max_iter=100
    )

    baseline_score = results["baseline"]
    print(f"Final Validation Performance: {baseline_score}")

    print("\nAblation results:")
    for name, score in results.items():
        delta = score - baseline_score
        sign = "+" if delta >= 0 else ""
        print(f"Ablation[{name}] RMSE: {score:.6f} | Delta vs baseline: {sign}{delta:.6f}")

    # Identify the most important part by the largest degradation when removed.
    degradations = {
        "subsampling": results["no_subsample"] - baseline_score,
        "vectorizer": results["no_vectorizer"] - baseline_score,
        "more_trees": results["fewer_trees"] - baseline_score,
    }
    most_important = max(degradations.items(), key=lambda kv: kv[1])
    print(
        f"\nMost contributing part: {most_important[0]} "
        f"(largest RMSE increase: {most_important[1]:.6f})"
    )


if __name__ == "__main__":
    main()
