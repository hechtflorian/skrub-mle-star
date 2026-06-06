
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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


def fit_predict_solution1(train_df, valid_df, test_df, target_col):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    data = skrub.var("data", train_df)
    X = data.drop(columns=[target_col]).skb.mark_as_X()
    y_node = data[target_col].skb.mark_as_y()

    try:
        vectorizer = skrub.TableVectorizer()
        learner_graph = X.skb.apply(vectorizer).skb.apply(
            HistGradientBoostingRegressor(random_state=42), y=y_node
        )
        learner = learner_graph.skb.make_learner(fitted=True)

        valid_pred = np.asarray(learner.predict({"data": valid_df})).ravel()
        test_pred = np.asarray(learner.predict({"data": test_df})).ravel()
        return valid_pred, test_pred, None
    except Exception:
        pipe = build_fallback_pipeline(X_train)
        pipe.fit(X_train, y_train)
        valid_pred = np.asarray(pipe.predict(valid_df.drop(columns=[target_col], errors="ignore"))).ravel()
        test_pred = np.asarray(pipe.predict(test_df)).ravel()
        return valid_pred, test_pred, None


def fit_predict_solution2(train_df, valid_df, test_df, target_col):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    try:
        pipe = build_fallback_pipeline(X_train)
        pipe.fit(X_train, y_train)
        valid_pred = np.asarray(pipe.predict(valid_df.drop(columns=[target_col], errors="ignore"))).ravel()
        test_pred = np.asarray(pipe.predict(test_df)).ravel()
        return valid_pred, test_pred, None
    except Exception:
        return None, None, RuntimeError("Solution 2 failed")


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"
    X_df = train_df.drop(columns=[target_col])
    y = train_df[target_col].values

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42, shuffle=True
    )
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    valid_split = train_df.iloc[valid_idx].reset_index(drop=True)
    y_valid = valid_split[target_col].values

    # Fit both solutions on the same holdout split
    s1_valid_pred, s1_test_pred, _ = fit_predict_solution1(
        train_split, valid_split, test_df, target_col
    )
    s2_valid_pred, s2_test_pred, _ = fit_predict_solution2(
        train_split, valid_split, test_df, target_col
    )

    # Robust blending weights based on holdout RMSE
    if s1_valid_pred is None and s2_valid_pred is None:
        pipe = build_fallback_pipeline(X_df)
        pipe.fit(X_df, y)
        valid_pred = pipe.predict(X_df)
        final_validation_score = rmse(y, valid_pred)
        test_pred = pipe.predict(test_df)
    elif s1_valid_pred is None:
        final_validation_score = rmse(y_valid, s2_valid_pred)
        test_pred = s2_test_pred
    elif s2_valid_pred is None:
        final_validation_score = rmse(y_valid, s1_valid_pred)
        test_pred = s1_test_pred
    else:
        rmse1 = rmse(y_valid, s1_valid_pred)
        rmse2 = rmse(y_valid, s2_valid_pred)

        if abs(rmse1 - rmse2) < 1e-6:
            w1, w2 = 0.5, 0.5
        else:
            inv1 = 1.0 / max(rmse1, 1e-12)
            inv2 = 1.0 / max(rmse2, 1e-12)
            w1 = inv1 / (inv1 + inv2)
            w2 = inv2 / (inv1 + inv2)

        blended_valid = w1 * s1_valid_pred + w2 * s2_valid_pred
        final_validation_score = rmse(y_valid, blended_valid)
        test_pred = w1 * s1_test_pred + w2 * s2_test_pred

    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
