
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Keep a DataOps-first workflow, with optional subsampling preserved for fast iteration.
    data = skrub.var("data", train_df)
    data = data.skb.subsample(n=min(8000, len(train_df)))

    target_col = "median_house_value"

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Simple, robust preprocessing in DataOps graph.
    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    # Fix: choose_float(log=True) requires low > 0.
    # Use a small positive lower bound instead of 0.0.
    hgb_shallow = HistGradientBoostingRegressor(
        learning_rate=skrub.choose_float(0.01, 0.2, log=True, name="shallow_learning_rate"),
        max_depth=skrub.choose_int(2, 6, name="shallow_max_depth"),
        min_samples_leaf=skrub.choose_int(10, 50, name="shallow_min_samples_leaf"),
        l2_regularization=skrub.choose_float(1e-8, 1.0, log=True, name="shallow_l2"),
        max_iter=skrub.choose_int(100, 400, name="shallow_max_iter"),
        random_state=42,
    )

    pred = X_vec.skb.apply(hgb_shallow, y=y)

    # DataOps evaluation
    cv_results = pred.skb.cross_validate(keep_subsampling=True)
    if isinstance(cv_results, dict):
        final_validation_score = cv_results.get("test_score", cv_results.get("score", np.nan))
        if isinstance(final_validation_score, (list, tuple, np.ndarray, pd.Series)):
            final_validation_score = float(np.mean(final_validation_score))
    else:
        try:
            final_validation_score = float(np.mean(cv_results["test_score"]))
        except Exception:
            final_validation_score = float("nan")

    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full subsampled DataOps graph and predict on test.
    learner = pred.skb.make_learner(fitted=True)
    test_preds = learner.predict({"data": test_df})

    # Ensure output format is exactly one column.
    if isinstance(test_preds, pd.DataFrame):
        out = test_preds.iloc[:, 0]
    elif isinstance(test_preds, pd.Series):
        out = test_preds
    else:
        out = pd.Series(np.asarray(test_preds).ravel())

    submission = pd.DataFrame({"median_house_value": out})
    submission.to_csv("submission.csv", index=False)

    # Optional local RMSE on training predictions if available via cross-validated prediction path not used here.
    return submission


if __name__ == "__main__":
    main()
