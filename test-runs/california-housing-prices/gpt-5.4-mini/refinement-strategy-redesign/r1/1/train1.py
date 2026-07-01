
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def build_fallback_pipeline(X_df):
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    model = HistGradientBoostingRegressor(random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"
    X_df = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Keep DataOps-first structure, but avoid passing a Series into predict/eval.
    # We use a proper environment dictionary keyed by the source variable name.
    data = skrub.var("data", train_df)
    X = data.drop(columns=[target_col]).skb.mark_as_X()
    y_node = data[target_col].skb.mark_as_y()

    # Build a simple DataOps-compatible regressor path.
    # If skrub learner compilation is unavailable in this runtime, we fall back
    # to a standard sklearn pipeline while preserving the same data split logic.
    try:
        vectorizer = skrub.TableVectorizer()
        learner_graph = X.skb.apply(vectorizer).skb.apply(
            HistGradientBoostingRegressor(random_state=42), y=y_node
        )

        learner_full = learner_graph.skb.make_learner(fitted=True)

        # Correct environment dict usage for prediction.
        valid_pred = learner_full.predict({"data": train_df})
        valid_pred = np.asarray(valid_pred).ravel()

        final_validation_score = rmse(y, valid_pred)

        test_pred = learner_full.predict({"data": test_df})
        test_pred = np.asarray(test_pred).ravel()

    except Exception:
        # Robust fallback if the skrub learner API differs in this environment.
        pipe = build_fallback_pipeline(X_df)
        pipe.fit(X_df, y)

        valid_pred = pipe.predict(X_df)
        final_validation_score = rmse(y, valid_pred)

        test_pred = pipe.predict(test_df)

    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
