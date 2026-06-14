
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

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

# Fill missing values with medians computed from the training features
median_values = X.median(numeric_only=True)
X_filled = X.fillna(median_values)
test_filled = test.fillna(median_values)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_filled, y, test_size=0.2, random_state=SEED
)

def fit_and_score(X_tr_, y_tr_, X_val_, y_val_, desc, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    params = {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 8,
        "loss_function": "RMSE",
        "random_seed": SEED,
        "verbose": False,
    }
    params.update(model_kwargs)

    train_pool = Pool(X_tr_, y_tr_)
    val_pool = Pool(X_val_, y_val_)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    preds = model.predict(X_val_)
    rmse = np.sqrt(mean_squared_error(y_val_, preds))
    print(f"{desc}: RMSE = {rmse:.5f}")
    return model, rmse

# Baseline
baseline_model, baseline_rmse = fit_and_score(X_tr, y_tr, X_val, y_val, "Baseline")

# Ablation 1: Disable missing-value imputation
X_zero_filled = X.fillna(0)
X_tr_1, X_val_1, y_tr_1, y_val_1 = train_test_split(
    X_zero_filled, y, test_size=0.2, random_state=SEED
)
ablation1_model, ablation1_rmse = fit_and_score(
    X_tr_1, y_tr_1, X_val_1, y_val_1, "Ablation 1 (zero-fill instead of median-fill)"
)

# Ablation 2: Change model depth from 8 to 4 without duplicate keyword arguments
ablation2_model, ablation2_rmse = fit_and_score(
    X_tr, y_tr, X_val, y_val, "Ablation 2 (depth=4 instead of depth=8)", model_kwargs={"depth": 4}
)

final_validation_score = baseline_rmse
print(f"Final Validation Performance: {final_validation_score}")

# Train final model on all training data and predict test set
final_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=False,
)
final_model.fit(Pool(X_filled, y), verbose=False)
test_preds = final_model.predict(test_filled)

submission = pd.DataFrame({"median_house_value": test_preds})
submission.to_csv("submission.csv", index=False)

print("\nAblation summary:")
print(f"Baseline RMSE: {baseline_rmse:.5f}")
print(f"Ablation 1 RMSE: {ablation1_rmse:.5f} | Delta: {ablation1_rmse - baseline_rmse:+.5f}")
print(f"Ablation 2 RMSE: {ablation2_rmse:.5f} | Delta: {ablation2_rmse - baseline_rmse:+.5f}")
