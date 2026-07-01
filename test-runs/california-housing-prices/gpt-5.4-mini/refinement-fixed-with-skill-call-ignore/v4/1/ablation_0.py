import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def load_data():
    train_path = os.path.join("./input", "train.csv")
    train_df = pd.read_csv(train_path)
    return train_df


def basic_cleaning(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def build_dataops_pipeline(train_part, target_col, model, use_vectorizer=True):
    data = skrub.var("data", train_part)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    if use_vectorizer:
        X = X.skb.apply(skrub.TableVectorizer())

    pred = X.skb.apply(model, y=y)
    return pred


def fit_and_validate_variant(train_df, valid_df, target_col, model, variant_name, use_vectorizer=True):
    train_part = train_df.reset_index(drop=True)
    valid_part = valid_df.reset_index(drop=True)

    pred = build_dataops_pipeline(
        train_part=train_part,
        target_col=target_col,
        model=model,
        use_vectorizer=use_vectorizer,
    )

    learner = pred.skb.make_learner(fitted=True)
    valid_features = valid_part.drop(columns=[target_col], errors="ignore")
    if use_vectorizer:
        valid_pred = learner.predict({"data": valid_features})
    else:
        valid_pred = learner.predict({"data": valid_features})
    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return learner, rmse


def main():
    target_col = "median_house_value"
    train_df = load_data()
    train_df = basic_cleaning(train_df)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(train_df))
    valid_size = max(1, int(0.2 * len(train_df)))
    valid_idx = perm[:valid_size]
    train_idx = perm[valid_size:]

    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    valid_part = train_df.iloc[valid_idx].reset_index(drop=True)

    baseline_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        random_state=42,
    )

    # Ablation 1: simpler model capacity
    low_capacity_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        max_iter=150,
        random_state=42,
    )

    # Ablation 2: disable TableVectorizer and feed raw numeric features only
    numeric_only_cols = [
        c for c in train_df.columns if c != target_col and train_df[c].dtype.kind in "biufc"
    ]
    train_numeric = train_part[numeric_only_cols + [target_col]].copy()
    valid_numeric = valid_part[numeric_only_cols + [target_col]].copy()

    baseline_learner, baseline_rmse = fit_and_validate_variant(
        train_part, valid_part, target_col, baseline_model, "baseline", use_vectorizer=True
    )

    low_capacity_learner, low_capacity_rmse = fit_and_validate_variant(
        train_part, valid_part, target_col, low_capacity_model, "lower_capacity", use_vectorizer=True
    )

    numeric_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        random_state=42,
    )
    numeric_learner, numeric_rmse = fit_and_validate_variant(
        train_numeric, valid_numeric, target_col, numeric_model, "numeric_only_no_vectorizer", use_vectorizer=False
    )

    results = {
        "baseline": baseline_rmse,
        "lower_capacity": low_capacity_rmse,
        "numeric_only_no_vectorizer": numeric_rmse,
    }

    final_validation_score = min(results.values())
    print(f"Final Validation Performance: {final_validation_score}")

    best_variant = min(results, key=results.get)
    print(f"Best ablation variant: {best_variant} | RMSE: {results[best_variant]}")

    deltas = {k: v - baseline_rmse for k, v in results.items() if k != "baseline"}
    most_contributing = min(deltas, key=deltas.get) if deltas else "baseline"
    print(
        f"Most impactful change: {most_contributing} "
        f"(delta vs baseline: {deltas.get(most_contributing, 0.0)})"
    )


if __name__ == "__main__":
    main()