
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


@skrub.deferred
def add_ratio_features_drop_households(df):
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


@skrub.deferred
def add_ratio_features_keep_households(df):
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
    return out


def fit_variant(train_data, feature_func, seed, split_random_state=42):
    split_train_idx, split_valid_idx = train_test_split(
        np.arange(len(train_data)), test_size=0.2, random_state=split_random_state
    )
    split_train = train_data.iloc[split_train_idx].copy()
    split_valid = train_data.iloc[split_valid_idx].copy()

    data_train = skrub.var("data", split_train)
    data_train_fe = data_train.skb.apply_func(feature_func)
    X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=500,
        learning_rate=0.05,
        depth=8,
        random_seed=seed,
        verbose=0,
    )

    pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
    learner = pred.skb.make_learner(fitted=True)

    valid_data = skrub.var("data", split_valid)
    valid_data_fe = valid_data.skb.apply_func(feature_func)
    valid_pred = learner.predict({"data": split_valid})

    rmse = mean_squared_error(split_valid[target_col], valid_pred) ** 0.5
    return learner, rmse, valid_pred


# Base model: original ratio-features pipeline dropping households
learner1, rmse1, _ = fit_variant(
    train_part, add_ratio_features_drop_households, seed=42, split_random_state=42
)

# Diversity model: keep households while adding ratios
learner2, rmse2, _ = fit_variant(
    train_part, add_ratio_features_keep_households, seed=42, split_random_state=42
)

# Slight seed/split variation with the original preprocessing
learner3, rmse3, _ = fit_variant(
    train_part, add_ratio_features_drop_households, seed=99, split_random_state=7
)

valid_pred1 = learner1.predict({"data": valid_part})
valid_pred2 = learner2.predict({"data": valid_part})
valid_pred3 = learner3.predict({"data": valid_part})

rmses = np.array([
    mean_squared_error(valid_part[target_col], valid_pred1) ** 0.5,
    mean_squared_error(valid_part[target_col], valid_pred2) ** 0.5,
    mean_squared_error(valid_part[target_col], valid_pred3) ** 0.5,
])

preds = np.vstack([valid_pred1, valid_pred2, valid_pred3]).T
weights = 1.0 / np.maximum(rmses, 1e-12)
weights = weights / weights.sum()
ensemble_valid_pred = np.sum(preds * weights.reshape(1, -1), axis=1)

# If predictions are very similar, use rank averaging as fallback
corr = np.corrcoef(preds.T)
avg_corr = (corr[np.triu_indices_from(corr, 1)]).mean() if preds.shape[1] > 1 else 1.0
if avg_corr > 0.995:
    rank_preds = np.vstack([pd.Series(p).rank(method="average").to_numpy() for p in [valid_pred1, valid_pred2, valid_pred3]]).T
    ensemble_valid_pred = np.sum(rank_preds * weights.reshape(1, -1), axis=1)
    ensemble_valid_pred = pd.Series(ensemble_valid_pred).rank(method="average").to_numpy()

final_validation_score = mean_squared_error(valid_part[target_col], ensemble_valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
