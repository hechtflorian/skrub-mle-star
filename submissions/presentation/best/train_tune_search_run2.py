
import sys
import subprocess
import json

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    if "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        if "total_rooms" in out.columns:
            out["rooms_per_household"] = (
                out["total_rooms"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "total_bedrooms" in out.columns:
            out["bedrooms_per_household"] = (
                out["total_bedrooms"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "population" in out.columns:
            out["population_per_household"] = (
                out["population"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.drop(columns=["households"], errors="ignore")
    return out

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_ratio_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

variants = {
    "d7_lr0.03": dict(
        depth=7,
        learning_rate=0.03,
        iterations=200,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    ),
    "d8_lr0.03": dict(
        depth=8,
        learning_rate=0.03,
        iterations=200,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    ),
    "d8_lr0.05": dict(
        depth=8,
        learning_rate=0.05,
        iterations=200,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    ),
    "d9_lr0.05": dict(
        depth=9,
        learning_rate=0.05,
        iterations=200,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    ),
}

model = skrub.choose_from(
    {
        k: CatBoostRegressor(**v)
        for k, v in variants.items()
    },
    name="model_variant",
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=4, n_jobs=1, random_state=42, fitted=True
)
search.fit({"data": train_part})

valid_learner = search.best_learner_
valid_pred = valid_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

best_choice = search.results_.iloc[0]["model_variant"]
best_params = dict(variants[str(best_choice)])
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
