
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub", "-q"])
    import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET = "median_house_value"


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df



def build_and_evaluate(train_df):
    data = train_df.copy()

    # Keep subsampling if present, but do not rely on it for fitting the final learner.
    if len(data) > 5000:
        preview_data = data.sample(n=5000, random_state=RANDOM_STATE)
    else:
        preview_data = data

    # Single preview-based split for fast structural search and scoring.
    preview_train, preview_val = train_test_split(
        preview_data, test_size=0.2, random_state=RANDOM_STATE
    )

    # DataOps graph on preview split
    data_var = skrub.var("data", preview_train)
    X = data_var.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data_var[TARGET].skb.mark_as_y()

    # Tiny structural search: preprocessing choice + a couple of high-impact HGB params.
    learning_rate = skrub.choose_float(0.03, 0.15, log=True, name="learning_rate")
    min_samples_leaf = skrub.choose_from(
        {"10": 10, "20": 20, "30": 30}, name="min_samples_leaf"
    )

    vectorized_variant = skrub.TableVectorizer()
    passthrough_variant = skrub.choose_from(
        {
            "table_vectorizer": vectorized_variant,
            "passthrough": None,
        },
        name="preprocessing_variant",
    )

    if passthrough_variant is None:
        X_proc = X
    else:
        X_proc = X.skb.apply(passthrough_variant)

    regressor = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=8,
        max_iter=250,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )

    pred = X_proc.skb.apply(regressor, y=y)

    # Small randomized search over the in-graph structural choices.
    search = pred.skb.make_randomized_search(
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        random_state=RANDOM_STATE,
        fitted=True,
    )

    # Fit/search on the preview environment only.
    search.fit({"data": preview_train})

    # Score the selected structure on the preview validation split.
    best_learner = search.best_learner_
    val_preds = best_learner.predict({"data": preview_val})
    final_validation_score = rmse(preview_val[TARGET], val_preds)

    # Refit the exact selected learner on the full training environment.
    final_learner = search.best_learner_
    final_learner.fit({"data": data})

    print(f"Final Validation Performance: {final_validation_score}")
    return final_learner, final_validation_score



def fit_full_model(train_df):
    data = train_df.copy()

    data_var = skrub.var("data", data)
    X = data_var.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data_var[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred = X_vec.skb.apply(regressor, y=y)
    learner = pred.skb.make_learner(fitted=False)
    learner.fit({"data": data})
    return learner


def main():
    train_df, test_df = load_data()

    _, final_validation_score = build_and_evaluate(train_df)

    final_learner = fit_full_model(train_df)
    test_preds = final_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_preds})
    submission.to_csv("submission.csv", index=False)

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
