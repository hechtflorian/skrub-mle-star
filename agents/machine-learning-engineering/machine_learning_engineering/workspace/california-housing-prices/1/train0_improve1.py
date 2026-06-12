
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Simple preprocessing and CatBoost model inside DataOps workflow
# CatBoost can handle raw numeric features directly; TableVectorizer keeps the pipeline DataOps-native.
vectorizer = skrub.TableVectorizer()

model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    random_seed=42,
    verbose=0,
)


import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Keep the same holdout protocol; no test refit here.
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

@skrub.deferred
def add_guarded_geo_features(df):
    out = df.copy()
    cols = set(out.columns)

    geo_pairs = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("x", "y"),
    ]

    for lat_name, lon_name in geo_pairs:
        if lat_name in cols and lon_name in cols:
            lat = pd.to_numeric(out[lat_name], errors="coerce")
            lon = pd.to_numeric(out[lon_name], errors="coerce")

            out[f"{lat_name}_abs"] = lat.abs()
            out[f"{lon_name}_abs"] = lon.abs()
            out[f"{lat_name}_{lon_name}_interaction"] = lat * lon
            out[f"{lat_name}_{lon_name}_radius"] = np.sqrt(lat ** 2 + lon ** 2)

            # Guarded proxy features: stable nonlinear spatial signal without heavy routing.
            lat_centered = lat - lat.median(skipna=True)
            lon_centered = lon - lon.median(skipna=True)
            out[f"{lat_name}_{lon_name}_centered_interaction"] = lat_centered * lon_centered
            out[f"{lat_name}_{lon_name}_abs_sum"] = lat.abs() + lon.abs()
            break

    return out

def build_and_score_variant(data_part, use_geo):
    data = skrub.var("data", data_part)
    if use_geo:
        data = data.skb.apply_func(add_guarded_geo_features)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    pred_graph = X.skb.apply(vectorizer).skb.apply(model, y=y)

    learner = pred_graph.skb.make_learner(fitted=True)
    pred = learner.predict({"data": valid_part})
    score = mean_squared_error(valid_part[target_col], pred) ** 0.5
    return score, pred_graph

baseline_score, _ = build_and_score_variant(train_part, use_geo=False)
geo_score, best_graph = build_and_score_variant(train_part, use_geo=True)

final_validation_score = baseline_score
best_variant = "baseline"

if geo_score < baseline_score:
    final_validation_score = geo_score
    best_variant = "baseline+geo"

print(f"Ablation[baseline] RMSE: {baseline_score}")
print(f"Ablation[baseline+geo] RMSE: {geo_score}")
print(f"Best ablation variant: {best_variant} | RMSE: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
