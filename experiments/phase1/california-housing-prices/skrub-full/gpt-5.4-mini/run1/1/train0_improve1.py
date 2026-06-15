

import os
import numpy as np
import pandas as pd
import skrub
import skrub.selectors as s
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Fix: catboost is unavailable in this environment, so use a sklearn-compatible fallback
# while keeping the same model family slot as a tree-based boosting regressor.
try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    from sklearn.ensemble import HistGradientBoostingRegressor as CatBoostRegressor

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

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

@skrub.deferred
def add_geo_and_ratio_features(df):
    out = df.copy()

    # Geo structure: cheap derived terms to help the tree model capture location patterns.
    if {"longitude", "latitude"}.issubset(out.columns):
        lon = out["longitude"]
        lat = out["latitude"]
        out["coord_radius"] = np.sqrt(lon**2 + lat**2)
        out["lat_lon_interaction"] = lat * lon
        out["abs_longitude"] = lon.abs()
        out["abs_latitude"] = lat.abs()

    # Keep the ratio features from the earlier structural attempt, but in one guarded block.
    if {"total_rooms", "households"}.issubset(out.columns):
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (
            (out["total_rooms"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    if {"total_bedrooms", "households"}.issubset(out.columns):
        denom = out["households"].replace(0, np.nan)
        out["bedrooms_per_household"] = (
            (out["total_bedrooms"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    if {"population", "households"}.issubset(out.columns):
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (
            (out["population"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    return out

# Post-FE graph: route the redundant raw column drop against the enriched feature set.
# This keeps a single resolved path and compares the safer one-column drop choice.
X_train_fe = X_train.skb.apply_func(add_geo_and_ratio_features)
X_train_full_raw = X_train_fe
X_train_drop_households = X_train_fe.skb.apply(skrub.DropCols(cols=s.cols("households")))
X_train_drop_bedrooms = X_train_fe.skb.apply(skrub.DropCols(cols=s.cols("total_bedrooms")))

# Select a single resolved route directly: drop only the safer redundant raw column.
# The geo + ratio block remains in both branches; we keep the more informative raw set.
X_train_routed = X_train_drop_households

vectorizer = skrub.TableVectorizer()

# Keep CatBoost-style parameters when available; fallback ignores unsupported params safely.
try:
    model = CatBoostRegressor(
        loss_function="RMSE",
        verbose=0,
        random_seed=42,
        iterations=500,
        learning_rate=0.05,
        depth=8,
    )
except TypeError:
    model = CatBoostRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        random_state=42,
    )

pred = X_train_routed.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
