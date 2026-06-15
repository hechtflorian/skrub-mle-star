
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

def add_ratio_features_drop(df):
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

def add_ratio_features_keep(df):
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

def fit_member(train_part, valid_part, feature_func, seed=42):
    data_train = skrub.var("data", train_part)
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

    valid_pred = learner.predict({"data": valid_part})
    test_pred = learner.predict({"data": test_df})
    return np.asarray(valid_pred), np.asarray(test_pred)

pred_a_valid, pred_a_test = fit_member(train_part, valid_part, add_ratio_features_drop, seed=42)
pred_b_valid, pred_b_test = fit_member(train_part, valid_part, add_ratio_features_keep, seed=43)

y_valid = valid_part[target_col].to_numpy()

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# Stage 1: optimize convex blend weight on validation
grid = np.linspace(0.0, 1.0, 1001)
best_w = 0.5
best_score = float("inf")
for w in grid:
    blend = w * pred_a_valid + (1.0 - w) * pred_b_valid
    score = rmse(y_valid, blend)
    if score < best_score:
        best_score = score
        best_w = float(w)

# Stage 2: tiny residual-correction/local search around best weight
local_grid = np.clip(np.linspace(best_w - 0.05, best_w + 0.05, 201), 0.0, 1.0)
for w in local_grid:
    blend = w * pred_a_valid + (1.0 - w) * pred_b_valid
    score = rmse(y_valid, blend)
    if score < best_score:
        best_score = score
        best_w = float(w)

# Optional soft-rank blend if members are extremely close
corr = np.corrcoef(pred_a_valid, pred_b_valid)[0, 1] if len(pred_a_valid) > 1 else 1.0
if np.isfinite(corr) and corr > 0.995:
    rank_a = pd.Series(pred_a_valid).rank(method="average").to_numpy()
    rank_b = pd.Series(pred_b_valid).rank(method="average").to_numpy()
    rank_best_w = best_w
    rank_best_score = float("inf")
    for w in grid:
        blend_rank = w * rank_a + (1.0 - w) * rank_b
        score = rmse(y_valid, blend_rank)
        if score < rank_best_score:
            rank_best_score = score
            rank_best_w = float(w)
    # map same weight rule to test ranks
    rank_a_test = pd.Series(pred_a_test).rank(method="average").to_numpy()
    rank_b_test = pd.Series(pred_b_test).rank(method="average").to_numpy()
    valid_blend = rank_best_w * rank_a + (1.0 - rank_best_w) * rank_b
    test_blend = rank_best_w * rank_a_test + (1.0 - rank_best_w) * rank_b_test
    final_validation_score = rmse(y_valid, valid_blend)
else:
    # small shrink toward 0.5 if improvement is tiny
    blend_valid = best_w * pred_a_valid + (1.0 - best_w) * pred_b_valid
    base_score = rmse(y_valid, blend_valid)
    individual_best = min(rmse(y_valid, pred_a_valid), rmse(y_valid, pred_b_valid))
    improvement = individual_best - base_score
    if improvement < 1e-4:
        best_w = 0.75 * best_w + 0.25 * 0.5
        blend_valid = best_w * pred_a_valid + (1.0 - best_w) * pred_b_valid
        base_score = rmse(y_valid, blend_valid)
    test_blend = best_w * pred_a_test + (1.0 - best_w) * pred_b_test
    final_validation_score = base_score

print(f"Final Validation Performance: {final_validation_score}")
