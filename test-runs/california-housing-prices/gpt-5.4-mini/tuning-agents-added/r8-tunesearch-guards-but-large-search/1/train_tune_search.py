
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

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

variants = {
    "d7_lr0.03_l24": dict(depth=7, learning_rate=0.03, l2_leaf_reg=4.0),
    "d8_lr0.03_l24": dict(depth=8, learning_rate=0.03, l2_leaf_reg=4.0),
    "d8_lr0.04_l24": dict(depth=8, learning_rate=0.04, l2_leaf_reg=4.0),
    "d9_lr0.04_l26": dict(depth=9, learning_rate=0.04, l2_leaf_reg=6.0),
}

model_variant = skrub.choose_from(
    {
        k: CatBoostRegressor(
            loss_function="RMSE",
            iterations=4500,
            random_seed=42,
            verbose=0,
            **params,
        )
        for k, params in variants.items()
    },
    name="model_variant",
)

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_housing_ratio_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

pred = X_train.skb.apply(model_variant, y=y_train)
search = pred.skb.make_randomized_search(n_iter=4, n_jobs=2, random_state=42, fitted=True)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

best_variant_name = search.best_params_.get("model_variant", None)
if hasattr(best_variant_name, "item"):
    best_variant_name = best_variant_name.item()
best_params = dict(variants[str(best_variant_name)]) if best_variant_name is not None else {}
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
