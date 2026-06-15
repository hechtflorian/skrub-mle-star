
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

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]

# Much lighter setup to avoid timeout
seeds = [42, 52]
split_seeds = [42, 52]

param_grid = [
    {"depth": 6, "learning_rate": 0.05},
    {"depth": 8, "learning_rate": 0.05},
    {"depth": 6, "learning_rate": 0.1},
]

all_test_preds = []
all_weights = []
all_val_rmses = []

best_overall_rmse = float("inf")
best_overall_model_info = None

for run_idx, (seed, split_seed) in enumerate(zip(seeds, split_seeds), start=1):
    print(f"\n===== Ensemble Run {run_idx}/{len(seeds)} | seed={seed} | split_seed={split_seed} =====")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=split_seed
    )

    best_val_rmse = float("inf")
    best_params = None

    for params in param_grid:
        model = CatBoostRegressor(
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=1200,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=100
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False
        )

        val_pred = model.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, val_pred)
        print(
            f"Depth={params['depth']}, LR={params['learning_rate']}, "
            f"BestIter={model.get_best_iteration()}, Val RMSE={val_rmse}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = params.copy()

    if best_params is None:
        best_params = {"depth": 6, "learning_rate": 0.05}

    print(f"Best config for run {run_idx}: {best_params}, Validation RMSE: {best_val_rmse}")

    # Train once on the full data with the selected config; keep it lightweight
    final_model = CatBoostRegressor(
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        iterations=1200,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100
    )

    final_model.fit(X, y, verbose=False)

    final_val_pred = final_model.predict(X_val)
    final_validation_score = root_mean_squared_error(y_val, final_val_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    test_pred = final_model.predict(test)

    all_test_preds.append(test_pred)
    all_val_rmses.append(final_validation_score)

    weight = 1.0 / max(final_validation_score, 1e-12)
    all_weights.append(weight)

    if final_validation_score < best_overall_rmse:
        best_overall_rmse = final_validation_score
        best_overall_model_info = {
            "run_idx": run_idx,
            "seed": seed,
            "split_seed": split_seed,
            "best_params": best_params,
            "val_rmse": final_validation_score,
        }

weights = np.array(all_weights, dtype=np.float64)
weights = weights / weights.sum()

ensemble_test_pred = np.zeros(len(test), dtype=np.float64)
for pred, w in zip(all_test_preds, weights):
    ensemble_test_pred += pred * w

ensemble_val_rmse = float(np.mean(all_val_rmses))
print(f"\nBest individual model info: {best_overall_model_info}")
print(f"Ensemble Mean of Individual Validation RMSEs: {ensemble_val_rmse}")
print(f"Final Validation Performance: {best_overall_rmse}")

submission = pd.DataFrame({"median_house_value": ensemble_test_pred})
submission.to_csv("submission.csv", index=False)
