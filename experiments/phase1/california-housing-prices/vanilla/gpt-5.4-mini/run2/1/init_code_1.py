
# Install required package if missing
import sys
import subprocess

def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

ensure_package("catboost", "catboost")

import os
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

# CatBoostRegressor model
model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200
)

# Fit with validation set and early best model selection
model.fit(
    X_tr,
    y_tr,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# Validation metric
val_pred = model.predict(X_val)
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"Final Validation Performance: {val_rmse}")

# Train on full data for test prediction
best_iter = model.get_best_iteration()
if best_iter is None or best_iter <= 0:
    best_iter = 3000

final_model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=best_iter,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200
)
final_model.fit(X, y, verbose=200)

test_pred = final_model.predict(test)

# Save submission
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
