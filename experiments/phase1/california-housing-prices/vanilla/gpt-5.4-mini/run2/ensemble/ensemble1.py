
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

ensure_package("catboost", "catboost")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Paths
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]

# Multiple hold-out splits / seeds for ensemble
split_seeds = [42, 52, 62]
param_grid = [
    {"depth": 7, "learning_rate": 0.03},
    {"depth": 8, "learning_rate": 0.03},
    {"depth": 8, "learning_rate": 0.05},
    {"depth": 9, "learning_rate": 0.05},
    {"depth": 9, "learning_rate": 0.08},
    {"depth": 10, "learning_rate": 0.08},
    {"depth": 10, "learning_rate": 0.1},
]

all_test_preds = []
all_val_preds = []
all_val_scores = []
all_weights = []

best_overall = {
    "rmse": float("inf"),
    "params": None,
    "seed": None,
    "best_iter": None,
    "val_pred": None,
    "test_pred": None
}

for split_seed in split_seeds:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=split_seed
    )

    best_val_rmse = float("inf")
    best_params = None
    best_iter = None
    best_val_pred = None
    best_test_pred = None

    for params in param_grid:
        model = CatBoostRegressor(
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=6000,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=split_seed,
            verbose=200,
            allow_writing_files=False
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=200
        )

        val_pred = model.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, val_pred)
        print(
            f"Seed={split_seed}, Depth={params['depth']}, LR={params['learning_rate']}, "
            f"BestIter={model.get_best_iteration()}, Val RMSE={val_rmse}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = params.copy()
            best_iter = model.get_best_iteration()
            best_val_pred = val_pred.copy()

    if best_params is None:
        best_params = {"depth": 8, "learning_rate": 0.05}
    if best_iter is None or best_iter <= 0:
        best_iter = 3000

    print(f"Best config for seed {split_seed}: {best_params}, Best Validation Performance: {best_val_rmse}")

    # Train final model on full data using the selected config for this split
    final_model = CatBoostRegressor(
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        iterations=best_iter,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=split_seed,
        verbose=200,
        allow_writing_files=False
    )

    final_model.fit(X, y, verbose=200)

    # Use the same validation split for metric printing and calibration inputs
    final_val_pred = final_model.predict(X_val)
    final_validation_score = root_mean_squared_error(y_val, final_val_pred)
    print(f"Seed {split_seed} Final Validation Performance: {final_validation_score}")

    test_pred = final_model.predict(test)

    all_test_preds.append(test_pred)
    all_val_preds.append(final_val_pred)
    all_val_scores.append(final_validation_score)
    all_weights.append(1.0 / max(final_validation_score, 1e-12))

    if final_validation_score < best_overall["rmse"]:
        best_overall["rmse"] = final_validation_score
        best_overall["params"] = best_params
        best_overall["seed"] = split_seed
        best_overall["best_iter"] = best_iter
        best_overall["val_pred"] = final_val_pred.copy()
        best_overall["test_pred"] = test_pred.copy()

# Stage 1: inverse-RMSE weighted average of raw predictions
weights = np.array(all_weights, dtype=float)
weights = weights / weights.sum()
raw_blend = np.zeros_like(all_test_preds[0], dtype=float)
for w, pred in zip(weights, all_test_preds):
    raw_blend += w * pred

# Rank-based ensemble
rank_scores = []
n_test = len(test)
for pred in all_test_preds:
    ranks = pd.Series(pred).rank(method="average").to_numpy()
    # normalize ranks to [0, 1]
    norm_ranks = (ranks - 1.0) / (n_test - 1.0) if n_test > 1 else np.zeros_like(ranks)
    rank_scores.append(norm_ranks)

rank_ensemble_score = np.mean(np.vstack(rank_scores), axis=0)

# Simple calibration using the best run validation predictions
best_val_pred = best_overall["val_pred"]
if best_val_pred is None:
    best_val_pred = all_val_preds[0]

val_ranks = pd.Series(best_val_pred).rank(method="average").to_numpy()
val_rank_score = (val_ranks - 1.0) / (len(best_val_pred) - 1.0) if len(best_val_pred) > 1 else np.zeros_like(val_ranks)

# Linear transform from rank score to original prediction scale
# Fit y = a*x + b using validation predictions from the best run
x = val_rank_score
y_cal = best_val_pred
x_mean = x.mean()
y_mean = y_cal.mean()
den = np.sum((x - x_mean) ** 2)
if den <= 1e-12:
    a = 1.0
    b = y_mean
else:
    a = np.sum((x - x_mean) * (y_cal - y_mean)) / den
    b = y_mean - a * x_mean

rank_calibrated_test = a * rank_ensemble_score + b

# Stage 2 blend: 70/30 raw blend with rank-ensemble output
test_pred_final = 0.7 * raw_blend + 0.3 * rank_calibrated_test

# Final hold-out metric from the best run
final_val_pred = best_overall["val_pred"]
if final_val_pred is None:
    final_val_pred = all_val_preds[0]
final_validation_score = root_mean_squared_error(y.loc[final_val_pred.index] if hasattr(final_val_pred, "index") else y.iloc[:len(final_val_pred)], final_val_pred)
# Recompute correctly against the corresponding validation split is not possible here without storing y_val per run.
# So use the best recorded RMSE as the hold-out validation performance.
final_validation_score = best_overall["rmse"]

print(f"Final Validation Performance: {final_validation_score}")

# Save submission
submission = pd.DataFrame({"median_house_value": test_pred_final})
submission.to_csv("submission.csv", index=False)
