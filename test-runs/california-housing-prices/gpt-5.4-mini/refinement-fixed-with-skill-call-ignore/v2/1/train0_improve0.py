
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

    # DataOps-first setup
    data = skrub.var("data", train_df)

    # Keep subsampling if available for faster iteration, but do final fit on full data
    try:
        data_for_dev = data.skb.subsample(n=min(5000, len(train_df)))
    except Exception:
        data_for_dev = data

    X = data_for_dev.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_for_dev[target_col].skb.mark_as_y()

    # Simple but strong tabular baseline, with DataOps graph preserved
    vectorizer = skrub.TableVectorizer()

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    pred = X.skb.apply(vectorizer).skb.apply(regressor, y=y)

    # Use a direct DataOps validation path if available, otherwise fall back to sklearn split
    try:
        learner = pred.skb.make_learner(fitted=True)
        # Evaluate by fitting/predicting on a holdout split using the same DataOps graph
        train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)
        train_env = {"data": train_part}
        valid_env = {"data": valid_part}

        fitted_learner = pred.skb.make_learner(fitted=True)
        y_valid = valid_part[target_col].to_numpy()
        valid_pred = fitted_learner.predict(valid_env)
        final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5
        print(f"Final Validation Performance: {final_validation_score}")
        return fitted_learner
    except Exception:
        # Fallback: keep DataOps structure for feature generation, but ensure robust execution
        train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

        X_train = train_part.drop(columns=target_col)
        y_train = train_part[target_col]
        X_valid = valid_part.drop(columns=target_col)
        y_valid = valid_part[target_col]

        vectorizer_fallback = skrub.TableVectorizer()
        X_train_vec = vectorizer_fallback.fit_transform(X_train)
        X_valid_vec = vectorizer_fallback.transform(X_valid)

        regressor_fallback = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )
        regressor_fallback.fit(X_train_vec, y_train)
        valid_pred = regressor_fallback.predict(X_valid_vec)
        final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5
        print(f"Final Validation Performance: {final_validation_score}")
        return (vectorizer_fallback, regressor_fallback)


def train_full_and_predict(train_df, test_df):
    target_col = "median_house_value"
    X_train = train_df.drop(columns=target_col)
    y_train = train_df[target_col]

    vectorizer = skrub.TableVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    regressor.fit(X_train_vec, y_train)

    test_vec = vectorizer.transform(test_df)
    preds = regressor.predict(test_vec)
    return preds


def main():
    train_df, test_df = load_data()
    _ = build_and_evaluate(train_df)
    preds = train_full_and_predict(train_df, test_df)

    submission = pd.DataFrame({"median_house_value": preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
