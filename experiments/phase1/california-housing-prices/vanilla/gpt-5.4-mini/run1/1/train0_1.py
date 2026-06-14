
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

# Ensure required packages are available
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost", "lightgbm"])

from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "median_house_value"
X = train.drop(columns=[target_col]).copy()
y = train[target_col].copy()

# Basic missing value handling
median_values = X.median(numeric_only=True)
X = X.fillna(median_values)
test = test.fillna(median_values)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# CatBoost model
train_pool = Pool(X_tr, y_tr)
val_pool = Pool(X_val, y_val)

cat_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=200,
)

cat_model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

cat_val_preds = cat_model.predict(X_val)
cat_test_preds = cat_model.predict(test)

# LightGBM model
lgb_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    objective="regression",
)

lgb_model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
)

lgb_val_preds = lgb_model.predict(X_val)
lgb_test_preds = lgb_model.predict(test)

# Simple ensemble
val_preds = 0.5 * cat_val_preds + 0.5 * lgb_val_preds
test_preds = 0.5 * cat_test_preds + 0.5 * lgb_test_preds

val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

submission = pd.DataFrame({target_col: test_preds})
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {val_rmse}")
