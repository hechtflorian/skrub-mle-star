import os
import warnings

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    train_df = pd.read_csv(train_path)
    return train_df


def make_learner(train_df, target_col, use_vectorizer=True, use_catboost=True):
    data = skrub.var("data", train_df)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    if use_vectorizer:
        X_proc = X.skb.apply(skrub.TableVectorizer())
    else:
        X_proc = X

    if use_catboost:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=1000,
            depth=8,
            learning_rate=0.03,
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
        )
    else:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=300,
            depth=6,
            learning_rate=0.05,
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
        )

    learner_plan = X_proc.skb.apply(model, y=y)
    learner = learner_plan.skb.make_learner(fitted=True)
    return learner


def evaluate_variant(train_df, valid_df, target_col, variant_name, use_vectorizer=True, use_catboost=True):
    learner = make_learner(
        train_df=train_df,
        target_col=target_col,
        use_vectorizer=use_vectorizer,
        use_catboost=use_catboost,
    )

    X_valid = valid_df.drop(columns=[target_col], errors="ignore")
    y_valid = valid_df[target_col].values
    valid_preds = learner.predict({"data": X_valid})
    rmse = mean_squared_error(y_valid, valid_preds) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return rmse


def main():
    train_df = load_data()
    target_col = "median_house_value"

    train_split, valid_split = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    results = {}

    # Baseline: original structure
    results["baseline_vectorizer_catboost"] = evaluate_variant(
        train_df=train_split,
        valid_df=valid_split,
        target_col=target_col,
        variant_name="baseline_vectorizer_catboost",
        use_vectorizer=True,
        use_catboost=True,
    )

    # Ablation 1: disable TableVectorizer, keep same model
    results["no_vectorizer_catboost"] = evaluate_variant(
        train_df=train_split,
        valid_df=valid_split,
        target_col=target_col,
        variant_name="no_vectorizer_catboost",
        use_vectorizer=False,
        use_catboost=True,
    )

    # Ablation 2: keep vectorizer, use a lighter CatBoost configuration
    results["vectorizer_light_catboost"] = evaluate_variant(
        train_df=train_split,
        valid_df=valid_split,
        target_col=target_col,
        variant_name="vectorizer_light_catboost",
        use_vectorizer=True,
        use_catboost=False,
    )

    best_variant = min(results, key=results.get)
    best_score = results[best_variant]

    print(f"Final Validation Performance: {results['baseline_vectorizer_catboost']}")
    print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")

    # Compare contribution: improvement over worst variant and relative deltas
    baseline = results["baseline_vectorizer_catboost"]
    no_vectorizer = results["no_vectorizer_catboost"]
    light_catboost = results["vectorizer_light_catboost"]

    delta_no_vectorizer = no_vectorizer - baseline
    delta_light_catboost = light_catboost - baseline

    print(f"Impact of disabling TableVectorizer (RMSE change): {delta_no_vectorizer}")
    print(f"Impact of changing CatBoost configuration (RMSE change): {delta_light_catboost}")

    if delta_no_vectorizer > delta_light_catboost:
        print("Most contributing part: TableVectorizer")
    elif delta_light_catboost > delta_no_vectorizer:
        print("Most contributing part: CatBoost configuration")
    else:
        print("Most contributing parts are tied or inconclusive on this validation split.")


if __name__ == "__main__":
    main()