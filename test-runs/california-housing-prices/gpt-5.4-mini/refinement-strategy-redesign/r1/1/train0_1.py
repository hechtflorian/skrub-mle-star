
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data(path):
    return pd.read_csv(path)


def build_and_evaluate(train_df, test_df=None):
    target_col = "median_house_value"

    # Robustly handle both labeled training data and unlabeled prediction data.
    # Use errors="ignore" so inference-time data without the target column does not fail.
    data = skrub.var("data", train_df)

    if target_col in train_df.columns:
        X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
        y = data[target_col].skb.mark_as_y()
    else:
        # Fallback in case this function is ever called on unlabeled data.
        X = data.skb.mark_as_X()
        y = None

    # Keep the DataOps-first structure while fixing the target-drop bug.
    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )

    pred = X_vec.skb.apply(model, y=y)

    # Validation split for a real validation score when labels are available.
    if target_col in train_df.columns and len(train_df) >= 10:
        train_part, valid_part = train_test_split(
            train_df, test_size=0.2, random_state=42
        )

        train_data = skrub.var("data", train_part)
        valid_data = skrub.var("data", valid_part)

        X_train = train_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
        y_train = train_data[target_col].skb.mark_as_y()

        X_valid = valid_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
        y_valid = valid_data[target_col].skb.mark_as_y()

        learner = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train).skb.make_learner(
            fitted=True
        )
        valid_pred = learner.predict({"data": valid_part})
        final_validation_score = mean_squared_error(
            valid_part[target_col].to_numpy(), np.asarray(valid_pred)
        ) ** 0.5
    else:
        final_validation_score = float("nan")

    print(f"Final Validation Performance: {final_validation_score}")
    return pred


def main():
    input_dir = "./input"
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")

    train_df = load_data(train_path)
    test_df = load_data(test_path) if os.path.exists(test_path) else None

    target_col = "median_house_value"

    # Build DataOps pipeline and fit on the available labeled training data.
    data = skrub.var("data", train_df)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
    learner = pred.skb.make_learner(fitted=True)

    if target_col in train_df.columns and len(train_df) >= 10:
        train_part, valid_part = train_test_split(
            train_df, test_size=0.2, random_state=42
        )
        # Refit on the training fold and validate on the holdout fold.
        fold_data = skrub.var("data", train_part)
        fold_X = fold_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
        fold_y = fold_data[target_col].skb.mark_as_y()
        fold_pred = fold_X.skb.apply(vectorizer).skb.apply(model, y=fold_y)
        fold_learner = fold_pred.skb.make_learner(fitted=True)

        valid_pred = fold_learner.predict({"data": valid_part})
        final_validation_score = mean_squared_error(
            valid_part[target_col].to_numpy(), np.asarray(valid_pred)
        ) ** 0.5
    else:
        final_validation_score = float("nan")

    print(f"Final Validation Performance: {final_validation_score}")

    if test_df is not None:
        test_pred = learner.predict({"data": test_df})
        submission = pd.DataFrame({"median_house_value": np.asarray(test_pred)})
        submission.to_csv("submission.csv", index=False)
    else:
        # No test file found; still emit a placeholder submission-like file from train median.
        median_value = float(train_df[target_col].median()) if target_col in train_df.columns else 0.0
        submission = pd.DataFrame(
            {"median_house_value": np.full(len(train_df), median_value)}
        )
        submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
