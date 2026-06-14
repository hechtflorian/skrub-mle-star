
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

# Ensure required package is available
try:
    from catboost import CatBoostRegressor, Pool
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

train_pool = Pool(X_tr, y_tr)
val_pool = Pool(X_val, y_val)

model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=200,
)

model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

val_preds = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

test_preds = model.predict(test)

submission = pd.DataFrame({target_col: test_preds})
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {val_rmse}")
