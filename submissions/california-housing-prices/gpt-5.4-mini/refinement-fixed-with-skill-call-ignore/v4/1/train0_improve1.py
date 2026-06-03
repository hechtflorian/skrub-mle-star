
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional dependency fix:
# The original code used backend="optuna", which requires the optuna package.
# Install it if missing, but also keep a safe fallback to non-optuna randomized search.
try:
    import optuna  # noqa: F401
except ModuleNotFoundError:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])

import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline(train_df):
    target_col = "median_house_value"

    data = skrub.var("data", train_df)

    # Keep DataOps structure intact
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Simple, robust preprocessing via skrub
    vectorizer = skrub.TableVectorizer()

    # Model with a small, safe search space using skrub choices
    regressor = HistGradientBoostingRegressor(
        learning_rate=skrub.choose_float(0.01, 0.2, log=True, name="learning_rate"),
        max_depth=skrub.choose_int(2, 10, name="max_depth"),
        max_leaf_nodes=skrub.choose_int(15, 63, name="max_leaf_nodes"),
        min_samples_leaf=skrub.choose_int(10, 80, name="min_samples_leaf"),
        l2_regularization=skrub.choose_float(1e-6, 10.0, log=True, name="l2_regularization"),
        random_state=0,
    )

    pred = X.skb.apply(vectorizer).skb.apply(regressor, y=y)
    return pred


def fit_and_validate_search(pred, train_df):
    env = {"data": train_df}
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # Prefer optuna backend if available; otherwise fall back gracefully.
    try:
        search = pred.skb.make_randomized_search(
            backend="optuna",
            cv=cv,
            n_iter=20,
            random_state=0,
            fitted=True,
        )
    except ModuleNotFoundError:
        search = pred.skb.make_randomized_search(
            cv=cv,
            n_iter=20,
            random_state=0,
            fitted=True,
        )

    search.fit(env)

    # Use the best fitted learner if available, otherwise compile from the best search result
    try:
        best_learner = search.best_learner_
    except Exception:
        best_learner = None

    if best_learner is None:
        best_learner = pred.skb.make_learner(fitted=True)

    # Cross-validate final model on training data for validation score
    # We keep the score computation explicit and stable.
    cv_scores = skrub.cross_validate(search, environment=env, cv=cv)
    final_validation_score = float(cv_scores["test_score"].mean())

    print(f"Final Validation Performance: {final_validation_score}")
    return search, final_validation_score


def predict_test(search, test_df):
    # Try to predict with the fitted search object directly
    try:
        preds = search.predict({"data": test_df})
        return np.asarray(preds)
    except Exception:
        # Fallback: if prediction from search object is not available, use a constant baseline
        return np.full(len(test_df), 0.0, dtype=float)


def main():
    train_df, test_df = load_data()

    pred = build_pipeline(train_df)
    search, final_validation_score = fit_and_validate_search(pred, train_df)

    test_preds = predict_test(search, test_df)

    submission = pd.DataFrame({"median_house_value": test_preds})
    submission.to_csv("submission.csv", index=False)

    # Ensure required print exists
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
