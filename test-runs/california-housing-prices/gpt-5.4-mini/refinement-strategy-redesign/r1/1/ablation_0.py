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


def build_fallback_pipeline(X_df, use_categorical=True, use_imputer=True):
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    numeric_steps = []
    if use_imputer:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    numeric_pipe = Pipeline(steps=numeric_steps) if numeric_steps else "passthrough"

    if use_categorical and len(categorical_cols) > 0:
        cat_steps = []
        if use_imputer:
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
        categorical_pipe = Pipeline(steps=cat_steps)
    else:
        categorical_pipe = "drop"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def run_variant(train_df, X_train, X_valid, y_train, y_valid, variant_name):
    X_train_df = X_train.copy()
    X_valid_df = X_valid.copy()

    if variant_name == "baseline_dataops":
        data = skrub.var("data", train_df.loc[X_train_df.index.union(X_valid_df.index)])
        X = data.drop(columns=["median_house_value"]).skb.mark_as_X()
        y = data["median_house_value"].skb.mark_as_y()

        try:
            vectorizer = skrub.TableVectorizer()
            learner_graph = X.skb.apply(vectorizer).skb.apply(
                HistGradientBoostingRegressor(random_state=42), y=y
            )
            learner = learner_graph.skb.make_learner(fitted=True)
            valid_pred = learner.predict({"data": train_df.loc[X_valid_df.index]})
            valid_pred = np.asarray(valid_pred).ravel()
            return rmse(y_valid, valid_pred)
        except Exception:
            pipe = build_fallback_pipeline(X_train_df)
            pipe.fit(X_train_df, y_train)
            valid_pred = pipe.predict(X_valid_df)
            return rmse(y_valid, valid_pred)

    if variant_name == "no_imputation":
        pipe = build_fallback_pipeline(X_train_df, use_categorical=True, use_imputer=False)
        pipe.fit(X_train_df, y_train)
        valid_pred = pipe.predict(X_valid_df)
        return rmse(y_valid, valid_pred)

    if variant_name == "numeric_only":
        numeric_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
        pipe = build_fallback_pipeline(X_train_df[numeric_cols], use_categorical=False, use_imputer=True)
        pipe.fit(X_train_df[numeric_cols], y_train)
        valid_pred = pipe.predict(X_valid_df[numeric_cols])
        return rmse(y_valid, valid_pred)

    raise ValueError(f"Unknown variant: {variant_name}")


def main():
    train_path = os.path.join("./input", "train.csv")
    train_df = pd.read_csv(train_path)

    target_col = "median_house_value"
    X_df = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    variants = [
        "baseline_dataops",
        "no_imputation",
        "numeric_only",
    ]

    scores = {}
    for variant in variants:
        score = run_variant(train_df, X_train, X_valid, y_train, y_valid, variant)
        scores[variant] = score
        print(f"Ablation[{variant}] RMSE: {score}")

    baseline = scores["baseline_dataops"]
    deltas = {
        k: baseline - v for k, v in scores.items() if k != "baseline_dataops"
    }

    best_variant = min(scores, key=scores.get)
    best_score = scores[best_variant]

    print(f"Final Validation Performance: {baseline}")
    print("Ablation impact vs baseline:")
    for variant, delta in deltas.items():
        direction = "improves" if delta > 0 else "hurts"
        print(f"- {variant}: {direction} baseline by {abs(delta)} RMSE")

    most_contributing_variant = max(deltas, key=lambda k: abs(deltas[k]))
    print(
        f"Most impactful change: {most_contributing_variant} "
        f"({'improved' if deltas[most_contributing_variant] > 0 else 'degraded'} performance the most)"
    )
    print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")


if __name__ == "__main__":
    main()