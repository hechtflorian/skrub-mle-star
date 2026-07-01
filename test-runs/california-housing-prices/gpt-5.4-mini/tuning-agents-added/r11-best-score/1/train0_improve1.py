

import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skrub import DropCols
import skrub.selectors as s

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
def add_geo_and_ratio_features(df):
    out = df.copy()
    cols = set(out.columns)

    # Geo feature block for latitude/longitude
    if {"latitude", "longitude"}.issubset(cols):
        lat = out["latitude"]
        lon = out["longitude"]
        out["coord_radius"] = np.sqrt(lat ** 2 + lon ** 2)
        out["lat_lon_product"] = lat * lon
        out["lat_abs"] = lat.abs()
        out["lon_abs"] = lon.abs()
        out["latitude_bin"] = pd.cut(lat, bins=8, labels=False, duplicates="drop")
        out["longitude_bin"] = pd.cut(lon, bins=8, labels=False, duplicates="drop")

    # Existing ratio features
    if {"total_rooms", "households"}.issubset(cols):
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (
            (out["total_rooms"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    if {"total_bedrooms", "total_rooms"}.issubset(cols):
        denom = out["total_rooms"].replace(0, np.nan)
        out["bedrooms_per_room"] = (
            (out["total_bedrooms"] / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    if {"population", "households"}.issubset(cols):
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (
            (out["population"] / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return out

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_geo_and_ratio_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

# Redundancy drop ablation: remove one collinear raw count after ratios are created
X_train = X_train.skb.apply(DropCols(cols=s.cols("households")))

# Post-FE vectorization on the full graph
vectorizer = skrub.TableVectorizer()

model = CatBoostRegressor(
    iterations=3000,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    verbose=0,
    random_seed=42,
    early_stopping_rounds=100,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

train_learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = train_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
