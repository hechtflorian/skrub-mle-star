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
    train_df = pd.read_csv(train_path)
    return train_df


def make_learner(train_df, use_vectorizer=True, use_hgb=True):
    data = skrub.var("data", train_df)
    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    X_used = X
    if use_vectorizer:
        vectorizer = skrub.TableVectorizer()
        X_used = X_used.skb.apply(vectorizer)

    if use_hgb:
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=8,
            max_iter=300,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )
    else:
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=8,
            max_iter=100,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )

    pred = X_used.skb.apply(regressor, y=y)
    learner = pred.skb.make_learner(fitted=False)
    return learner


def evaluate_variant(train_split, val_split, use_vectorizer=True, use_hgb=True):
    learner = make_learner(train_split, use_vectorizer=use_vectorizer, use_hgb=use_hgb)
    learner.fit({"data": train_split})
    val_preds = learner.predict({"data": val_split})
    score = rmse(val_split[TARGET], val_preds)
    return score


def main():
    train_df = load_data()

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_STATE
    )

    variants = {
        "baseline_vectorizer_hgb": {"use_vectorizer": True, "use_hgb": True},
        "no_vectorizer": {"use_vectorizer": False, "use_hgb": True},
        "lighter_hgb": {"use_vectorizer": True, "use_hgb": False},
    }

    results = {}
    for name, cfg in variants.items():
        score = evaluate_variant(train_split, val_split, **cfg)
        results[name] = score
        print(f"Ablation[{name}] RMSE: {score:.6f}")

    baseline = results["baseline_vectorizer_hgb"]
    for name, score in results.items():
        delta = score - baseline
        sign = "+" if delta >= 0 else "-"
        print(f"Effect vs baseline[{name}]: {sign}{abs(delta):.6f} RMSE")

    best_variant = min(results, key=results.get)
    best_score = results[best_variant]

    impact = {name: score - baseline for name, score in results.items() if name != "baseline_vectorizer_hgb"}
    most_contributing = max(impact, key=lambda k: abs(impact[k]))
    print(f"Most influential modification: {most_contributing} | delta RMSE: {impact[most_contributing]:.6f}")
    print(f"Best ablation variant: {best_variant} | RMSE: {best_score:.6f}")

    final_validation_score = baseline
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()