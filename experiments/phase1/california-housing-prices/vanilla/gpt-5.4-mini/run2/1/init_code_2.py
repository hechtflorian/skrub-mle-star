
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

# Ensure xgboost is available
try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBRegressor

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Paths
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]

# Hold-out validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# XGBoost Regressor
model = XGBRegressor(
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

# Fit without early_stopping_rounds for compatibility with installed XGBoost API
model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=200,
)

# Validation metric
val_pred = model.predict(X_val)
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"Final Validation Performance: {val_rmse}")

# Train final model on full data
final_model = XGBRegressor(
    n_estimators=model.n_estimators,
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

final_model.fit(X, y, verbose=False)

# Test predictions
test_pred = final_model.predict(test)

# Save submission
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
