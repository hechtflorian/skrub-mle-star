
import sys
import subprocess
import os
import numpy as np
import pandas as pd

def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

ensure_package("catboost", "catboost")

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Paths
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic feature/target split
X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]

# Hold-out validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CatBoostRegressor model
model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    od_type="Iter",
    od_wait=100,
    subsample=0.8,
    bootstrap_type="Bernoulli",
    verbose=200
)

# Fit model with validation
model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# Validation performance
val_pred = model.predict(X_val)
final_validation_score = root_mean_squared_error(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Train on full data for final predictions
final_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=best_iter if 'best_iter' in locals() else model.get_best_iteration() if model.get_best_iteration() is not None else 3000,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=200,
    subsample=0.8,
    bootstrap_type="Bernoulli"
)

# Determine best iteration safely
best_iteration = model.get_best_iteration()
if best_iteration is None or best_iteration <= 0:
    best_iteration = 3000

final_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=best_iteration,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=200,
    subsample=0.8,
    bootstrap_type="Bernoulli"
)

final_model.fit(X, y)

# Predict on test set
test_pred = final_model.predict(test)

# Save submission
submission = pd.DataFrame({
    "median_house_value": test_pred
})
submission.to_csv("submission.csv", index=False)
print(submission.head())
