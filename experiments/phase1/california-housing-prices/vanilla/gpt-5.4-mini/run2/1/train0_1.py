
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Ensure required packages are available
def ensure_package(pkg_name, import_name=None):
    __import__(import_name or pkg_name)

try:
    ensure_package("catboost", "catboost")
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    ensure_package("catboost", "catboost")

try:
    ensure_package("xgboost", "xgboost")
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    ensure_package("xgboost", "xgboost")

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Paths
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

# Reproducibility
SEED = 42

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]

# Hold-out validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# CatBoost model
cat_model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=SEED,
    verbose=200
)

cat_model.fit(
    X_tr,
    y_tr,
    eval_set=(X_val, y_val),
    use_best_model=True
)

cat_val_pred = cat_model.predict(X_val)

# XGBoost model
xgb_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=SEED,
    objective="reg:squarederror",
)

xgb_model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=200,
)

xgb_val_pred = xgb_model.predict(X_val)

# Simple ensemble on validation
val_pred = 0.5 * cat_val_pred + 0.5 * xgb_val_pred
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"Final Validation Performance: {val_rmse}")

# Train final CatBoost model on full data
best_iter = cat_model.get_best_iteration()
if best_iter is None or best_iter <= 0:
    best_iter = 3000

final_cat_model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=best_iter,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=SEED,
    verbose=200
)
final_cat_model.fit(X, y, verbose=200)

# Train final XGBoost model on full data
final_xgb_model = XGBRegressor(
    n_estimators=xgb_model.n_estimators,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=SEED,
    objective="reg:squarederror",
)
final_xgb_model.fit(X, y, verbose=False)

# Test predictions and ensemble
cat_test_pred = final_cat_model.predict(test)
xgb_test_pred = final_xgb_model.predict(test)
test_pred = 0.5 * cat_test_pred + 0.5 * xgb_test_pred

# Save submission
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
