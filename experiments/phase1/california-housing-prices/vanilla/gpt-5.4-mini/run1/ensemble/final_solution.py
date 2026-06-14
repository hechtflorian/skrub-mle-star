import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

# Ensure required package is available
from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")
final_dir = "./final"
os.makedirs(final_dir, exist_ok=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "median_house_value"
X = train.drop(columns=[target_col]).copy()
y = train[target_col].copy()

# Fill missing values with train medians
medians = X.median(numeric_only=True)
X = X.fillna(medians)
test = test.fillna(medians)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Simple, cheap feature engineering
    out["rooms_per_household"] = out["total_rooms"] / (out["households"] + 1e-6)
    out["bedrooms_per_room"] = out["total_bedrooms"] / (out["total_rooms"] + 1e-6)
    out["population_per_household"] = out["population"] / (out["households"] + 1e-6)
    out["rooms_per_person"] = out["total_rooms"] / (out["population"] + 1e-6)
    out["bedrooms_per_household"] = out["total_bedrooms"] / (out["households"] + 1e-6)
    out["income_x_age"] = out["median_income"] * out["housing_median_age"]
    out["lat_long_sum"] = out["latitude"] + out["longitude"]
    out["lat_long_diff"] = out["latitude"] - out["longitude"]
    out["income_sq"] = out["median_income"] ** 2
    out["age_sq"] = out["housing_median_age"] ** 2
    return out

X = add_features(X)
test = add_features(test)

# Align columns just in case
X, test = X.align(test, join="left", axis=1, fill_value=np.nan)
X = X.fillna(X.median(numeric_only=True))
test = test.fillna(X.median(numeric_only=True))

# Fast KFold CatBoost training
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
test_preds = np.zeros(len(test))

cat_params = dict(
    iterations=1200,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=SEED,
    eval_metric="RMSE",
    verbose=False,
    subsample=0.8,
    colsample_bylevel=0.8,
    l2_leaf_reg=3.0,
    od_type="Iter",
    od_wait=80,
)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_va, y_va)

    model = CatBoostRegressor(**cat_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    oof[va_idx] = model.predict(X_va)
    test_preds += model.predict(test) / kf.n_splits

final_validation_score = np.sqrt(mean_squared_error(y, oof))

submission = pd.DataFrame({target_col: test_preds})
submission.to_csv(os.path.join(final_dir, "submission.csv"), index=False)

print(f"Final Validation Performance: {final_validation_score}")