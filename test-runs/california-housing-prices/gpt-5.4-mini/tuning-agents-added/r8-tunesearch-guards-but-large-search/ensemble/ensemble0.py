
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

# --- Model 1: CatBoost pipeline (kept intact) ---
data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_housing_ratio_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

pred_graph = X_train.skb.apply(build_model(), y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)

valid_pred_cat = learner.predict({"data": valid_part})

# Full-train CatBoost for test predictions (submission-stage logic kept)
data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_housing_ratio_features)
X_full = data_full_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

full_pred_graph = X_full.skb.apply(build_model(), y=y_full)
full_learner = full_pred_graph.skb.make_learner(fitted=True)
test_pred_cat = full_learner.predict({"data": test_df})

# --- Model 2: lightweight additional model for ensemble ---
# Using the same feature engineering to keep compatibility and avoid touching the original pipeline.
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_hgb_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=8,
                max_iter=500,
                random_state=42,
                l2_regularization=0.0,
            )),
        ]
    )

data_train_hgb = skrub.var("data", train_part)
data_train_hgb_fe = data_train_hgb.skb.apply_func(add_housing_ratio_features)
X_train_hgb = data_train_hgb_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_hgb = data_train_hgb_fe[target_col].skb.mark_as_y()

hgb_graph = X_train_hgb.skb.apply(build_hgb_model(), y=y_train_hgb)
hgb_learner = hgb_graph.skb.make_learner(fitted=True)
valid_pred_hgb = hgb_learner.predict({"data": valid_part})

# Full-train HGB for test predictions
data_full_hgb = skrub.var("data", train_df)
data_full_hgb_fe = data_full_hgb.skb.apply_func(add_housing_ratio_features)
X_full_hgb = data_full_hgb_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_hgb = data_full_hgb_fe[target_col].skb.mark_as_y()

hgb_full_graph = X_full_hgb.skb.apply(build_hgb_model(), y=y_full_hgb)
hgb_full_learner = hgb_full_graph.skb.make_learner(fitted=True)
test_pred_hgb = hgb_full_learner.predict({"data": test_df})

# --- Thin ensemble layer ---
pred_df_valid = pd.DataFrame(
    {
        "pred_catboost": np.asarray(valid_pred_cat).reshape(-1),
        "pred_hgb": np.asarray(valid_pred_hgb).reshape(-1),
        "target": valid_part[target_col].to_numpy(),
    }
)

pred_df_test = pd.DataFrame(
    {
        "pred_catboost": np.asarray(test_pred_cat).reshape(-1),
        "pred_hgb": np.asarray(test_pred_hgb).reshape(-1),
    }
)

weights = np.linspace(0.0, 1.0, 11)
best_w = 0.5
best_rmse = float("inf")

# Try raw blend first
for w in weights:
    blended = w * pred_df_valid["pred_catboost"].to_numpy() + (1.0 - w) * pred_df_valid["pred_hgb"].to_numpy()
    rmse = mean_squared_error(pred_df_valid["target"], blended) ** 0.5
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

# Optional simple calibration if raw scales differ noticeably
cat_std = pred_df_valid["pred_catboost"].std(ddof=0) + 1e-12
hgb_std = pred_df_valid["pred_hgb"].std(ddof=0) + 1e-12
scale_ratio = max(cat_std, hgb_std) / min(cat_std, hgb_std)

if scale_ratio > 1.5:
    cat_mu = pred_df_valid["pred_catboost"].mean()
    hgb_mu = pred_df_valid["pred_hgb"].mean()
    cat_z_valid = (pred_df_valid["pred_catboost"] - cat_mu) / cat_std
    hgb_z_valid = (pred_df_valid["pred_hgb"] - hgb_mu) / hgb_std

    best_w_z = 0.5
    best_rmse_z = float("inf")
    for w in weights:
        blended_z = w * cat_z_valid.to_numpy() + (1.0 - w) * hgb_z_valid.to_numpy()
        rmse = mean_squared_error(pred_df_valid["target"], blended_z) ** 0.5
        if rmse < best_rmse_z:
            best_rmse_z = rmse
            best_w_z = w

    # Use z-space blend only if it helps
    if best_rmse_z < best_rmse:
        cat_z_test = (pred_df_test["pred_catboost"] - cat_mu) / cat_std
        hgb_z_test = (pred_df_test["pred_hgb"] - hgb_mu) / hgb_std
        final_test_pred_z = best_w_z * cat_z_test.to_numpy() + (1.0 - best_w_z) * hgb_z_test.to_numpy()
        # Map back approximately to target scale using blended validation target moments
        target_mu = pred_df_valid["target"].mean()
        target_std = pred_df_valid["target"].std(ddof=0) + 1e-12
        test_pred = final_test_pred_z * target_std + target_mu

        valid_pred_z = best_w_z * cat_z_valid.to_numpy() + (1.0 - best_w_z) * hgb_z_valid.to_numpy()
        valid_pred_for_metric = valid_pred_z * target_std + target_mu
        final_validation_score = mean_squared_error(pred_df_valid["target"], valid_pred_for_metric) ** 0.5
    else:
        test_pred = best_w * pred_df_test["pred_catboost"].to_numpy() + (1.0 - best_w) * pred_df_test["pred_hgb"].to_numpy()
        final_validation_score = best_rmse
else:
    test_pred = best_w * pred_df_test["pred_catboost"].to_numpy() + (1.0 - best_w) * pred_df_test["pred_hgb"].to_numpy()
    final_validation_score = best_rmse

print(f"Final Validation Performance: {final_validation_score}")

sub = pd.DataFrame({"median_house_value": test_pred})
sub.to_csv("submission.csv", index=False)
