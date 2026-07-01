
import os
import json
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

@skrub.deferred
def add_housing_ratio_features(df):
    out = df.copy()
    if "households" in out.columns and "total_rooms" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "total_rooms" in out.columns and "total_bedrooms" in out.columns:
        denom = out["total_rooms"].replace(0, np.nan)
        out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "households" in out.columns and "population" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (out["population"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    return out

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_housing_ratio_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

variant_specs = {
    "d6_lr0.03_it800": {
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 800,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    },
    "d8_lr0.03_it800": {
        "depth": 8,
        "learning_rate": 0.03,
        "iterations": 800,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    },
    "d8_lr0.05_it600": {
        "depth": 8,
        "learning_rate": 0.05,
        "iterations": 600,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    },
    "d10_lr0.03_it600": {
        "depth": 10,
        "learning_rate": 0.03,
        "iterations": 600,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    },
}

model = skrub.choose_from(
    {
        k: CatBoostRegressor(**params)
        for k, params in variant_specs.items()
    },
    name="model_variant",
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

search = pred_graph.skb.make_randomized_search(
    n_iter=4, n_jobs=2, random_state=42, fitted=True
)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

best_choice = None
if hasattr(search, "best_params_") and isinstance(search.best_params_, dict):
    for v in search.best_params_.values():
        if isinstance(v, str) and v in variant_specs:
            best_choice = v
            break

if best_choice is None and hasattr(search, "results_"):
    try:
        best_row = search.results_.iloc[0]
        if "model_variant" in best_row.index:
            best_choice = best_row["model_variant"]
    except Exception:
        pass

if best_choice is None:
    best_choice = "d8_lr0.03_it800"

best_params = dict(variant_specs[best_choice])
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
