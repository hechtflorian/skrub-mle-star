
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

# Hold-out validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = [
    {"depth": 7, "learning_rate": 0.03},
    {"depth": 8, "learning_rate": 0.03},
    {"depth": 8, "learning_rate": 0.05},
    {"depth": 9, "learning_rate": 0.05},
    {"depth": 9, "learning_rate": 0.08},
    {"depth": 10, "learning_rate": 0.08},
    {"depth": 10, "learning_rate": 0.1},
]

best_val_rmse = float("inf")
best_params = None
best_iter = None

for params in param_grid:
    model = CatBoostRegressor(
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        iterations=6000,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
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
        f"Depth={params['depth']}, LR={params['learning_rate']}, "
        f"BestIter={model.get_best_iteration()}, Val RMSE={val_rmse}"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params.copy()
        best_iter = model.get_best_iteration()

# Fallback in case something unexpected happens
if best_params is None:
    best_params = {"depth": 8, "learning_rate": 0.05}
if best_iter is None or best_iter <= 0:
    best_iter = 3000

print(f"Best config: {best_params}, Best Validation Performance: {best_val_rmse}")

# Train final model on full data
final_model = CatBoostRegressor(
    depth=best_params["depth"],
    learning_rate=best_params["learning_rate"],
    iterations=best_iter,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False
)

final_model.fit(X, y, verbose=200)

# Final validation performance for parsing
final_val_pred = final_model.predict(X_val)
final_validation_score = root_mean_squared_error(y_val, final_val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Predict test
test_pred = final_model.predict(test)

# Save submission
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
