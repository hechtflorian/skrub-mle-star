
import os
import json
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "median_house_value"

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Small variant grid for CatBoost (non-sklearn estimator: choose whole estimator objects)
model = skrub.choose_from(
    {
        "d6_lr0.03": CatBoostRegressor(
            depth=6,
            learning_rate=0.03,
            iterations=500,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        ),
        "d8_lr0.03": CatBoostRegressor(
            depth=8,
            learning_rate=0.03,
            iterations=500,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        ),
        "d8_lr0.05": CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            iterations=500,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        ),
        "d10_lr0.03": CatBoostRegressor(
            depth=10,
            learning_rate=0.03,
            iterations=500,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        ),
    },
    name="model_variant",
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=4,
    n_jobs=1,
    random_state=42,
    fitted=True,
)

search.fit({"data": train_part})
best_learner = search.best_learner_
valid_pred = best_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Optional: print best params in a JSON-safe way
best_params = {}
if hasattr(search, "best_params_") and search.best_params_ is not None:
    for k, v in search.best_params_.items():
        if hasattr(v, "item"):
            v = v.item()
        best_params[k] = v
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
