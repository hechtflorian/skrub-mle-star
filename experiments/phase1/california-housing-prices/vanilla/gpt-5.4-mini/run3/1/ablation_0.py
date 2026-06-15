import os
import math
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def train_and_eval(X_tr, y_tr, X_val, y_val, params, label):
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    pred = model.predict(X_val)
    score = rmse(y_val, pred)
    print(f"{label}: RMSE = {score:.6f}")
    return score

def main():
    train = pd.read_csv("./input/train.csv")

    X = train.drop(columns=["median_house_value"])
    y = train["median_house_value"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_params = dict(
        iterations=3000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "CPU"
    )

    print("Running ablation study...\n")

    # Baseline
    baseline_score = train_and_eval(X_tr, y_tr, X_val, y_val, base_params, "Baseline")

    # Ablation 1: fewer iterations
    ablation_iters = base_params.copy()
    ablation_iters["iterations"] = 500
    score_iters = train_and_eval(
        X_tr, y_tr, X_val, y_val, ablation_iters, "Ablation - fewer iterations"
    )

    # Ablation 2: shallower trees
    ablation_depth = base_params.copy()
    ablation_depth["depth"] = 4
    score_depth = train_and_eval(
        X_tr, y_tr, X_val, y_val, ablation_depth, "Ablation - shallower depth"
    )

    # Report contribution
    diffs = {
        "iterations": score_iters - baseline_score,
        "depth": score_depth - baseline_score,
    }

    worst_part = max(diffs, key=diffs.get)
    print("\nPerformance impact vs baseline:")
    for part, delta in diffs.items():
        sign = "+" if delta >= 0 else ""
        print(f"- {part}: {sign}{delta:.6f} RMSE")

    print(f"\nMost important part for performance: {worst_part}")

if __name__ == "__main__":
    main()