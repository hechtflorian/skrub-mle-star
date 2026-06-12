
import json
import os

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


@skrub.deferred
def add_house_feature_ratios(df):
    out = df.copy()
    if "total_rooms" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "population" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (out["population"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "total_bedrooms" in out.columns and "total_rooms" in out.columns:
        denom = out["total_rooms"].replace(0, np.nan)
        out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    return out


data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_house_feature_ratios)

X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

depth = skrub.choose_int(6, 10, default=8, name="depth")
learning_rate = skrub.choose_float(
    0.02, 0.06, default=0.03, log=False, name="learning_rate"
)
iterations = skrub.choose_int(2000, 4000, default=3000, name="iterations")

model = CatBoostRegressor(
    iterations=iterations,
    learning_rate=learning_rate,
    depth=depth,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=4, n_jobs=2, random_state=42, fitted=True
)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

best_params = {}
for key, value in search.best_params_.items():
    if hasattr(value, "item"):
        value = value.item()
    best_params[key] = value
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
