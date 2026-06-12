
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

train_path = "./input/train.csv"
test_path = "./input/test.csv"

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

    ratio_specs = [
        ("total_rooms", "households", "rooms_per_household"),
        ("total_bedrooms", "households", "bedrooms_per_household"),
        ("population", "households", "population_per_household"),
        ("total_rooms", "population", "rooms_per_person"),
        ("total_bedrooms", "total_rooms", "bedrooms_per_room"),
    ]

    for numer_col, denom_col, out_name in ratio_specs:
        if numer_col in out.columns and denom_col in out.columns:
            denom = out[denom_col].replace(0, np.nan)
            ratio = (out[numer_col] / denom).replace([np.inf, -np.inf], np.nan)
            out[out_name] = ratio.fillna(0.0)

    if "latitude" in out.columns and "longitude" in out.columns:
        out["lat_lon_product"] = out["latitude"] * out["longitude"]
        out["lat_abs"] = out["latitude"].abs()
        out["lon_abs"] = out["longitude"].abs()
        out["geo_radius"] = np.sqrt(out["latitude"] ** 2 + out["longitude"] ** 2)

    return out

def build_model():
    return CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.04,
        iterations=4500,
        random_seed=42,
        verbose=200,
        l2_leaf_reg=4.0,
    )

def build_linear_model():
    return Ridge(alpha=1.0, random_state=42)

# --- CatBoost skrub pipeline (kept intact) ---
data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_housing_ratio_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

pred_graph = X_train.skb.apply(build_model(), y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)

valid_pred_cat = learner.predict({"data": valid_part})

# --- Linear-style blend partner on the same skrub-transformed features ---
linear_graph = X_train.skb.apply(build_linear_model(), y=y_train)
linear_learner = linear_graph.skb.make_learner(fitted=True)
valid_pred_lin = linear_learner.predict({"data": valid_part})

# --- Tiny meta-model on base prediction columns only ---
meta_X_valid = np.column_stack([
    valid_pred_cat,
    valid_pred_lin,
    0.5 * (valid_pred_cat + valid_pred_lin),
])

meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(meta_X_valid, valid_part[target_col].values)

meta_valid_pred = meta_model.predict(meta_X_valid)

# Clip to a reasonable range from training target
y_min = train_df[target_col].min()
y_max = train_df[target_col].max()
meta_valid_pred = np.clip(meta_valid_pred, y_min, y_max)

final_validation_score = mean_squared_error(valid_part[target_col], meta_valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# --- Full-train refit for submission ---
data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_housing_ratio_features)
X_full = data_full_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

full_cat_graph = X_full.skb.apply(build_model(), y=y_full)
full_cat_learner = full_cat_graph.skb.make_learner(fitted=True)
test_pred_cat = full_cat_learner.predict({"data": test_df})

full_lin_graph = X_full.skb.apply(build_linear_model(), y=y_full)
full_lin_learner = full_lin_graph.skb.make_learner(fitted=True)
test_pred_lin = full_lin_learner.predict({"data": test_df})

meta_X_test = np.column_stack([
    test_pred_cat,
    test_pred_lin,
    0.5 * (test_pred_cat + test_pred_lin),
])

test_pred = meta_model.predict(meta_X_test)
test_pred = np.clip(test_pred, y_min, y_max)

os.makedirs("./final", exist_ok=True)
sub = pd.DataFrame({"median_house_value": test_pred})
sub.to_csv("./final/submission.csv", index=False)
