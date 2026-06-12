
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skrub import ApplyToCols, SquashingScaler

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
from skrub import ApplyToCols, SquashingScaler

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

@skrub.deferred
def add_guarded_housing_ratios(df):
    out = df.copy()

    if "total_rooms" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household_struct"] = (
            (out["total_rooms"] / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    if "population" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household_struct"] = (
            (out["population"] / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return out

def build_graph(dataframe, use_derived_features):
    data = skrub.var("data", dataframe)

    if use_derived_features:
        data = data.skb.apply_func(add_guarded_housing_ratios)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Lightweight preprocessing refinement: stabilize numeric scale on the post-FE graph
    X = X.skb.apply(ApplyToCols(SquashingScaler(max_absolute_value=3), cols=skrub.selectors.numeric()))

    vectorizer = skrub.TableVectorizer()
    pred_graph = X.skb.apply(vectorizer).skb.apply(model, y=y)
    return pred_graph

# Single bounded ablation: baseline vs guarded derived-feature block
variants = {
    "baseline": False,
    "derived_features": True,
}

best_variant = None
best_score = np.inf
best_alpha = 0.0
best_holdout_min = None
best_holdout_max = None

for variant_name, use_derived_features in variants.items():
    pred_graph = build_graph(train_part, use_derived_features)
    val_learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})

    # clip to observed holdout target range before fitting blend
    holdout_min = valid_part[target_col].min()
    holdout_max = valid_part[target_col].max()
    pred_1 = np.clip(np.asarray(valid_pred, dtype=float), holdout_min, holdout_max)

    # Build second pipeline prediction for residual-corrected blending
    # Keep pipeline structure intact; only use the alternative graph as a correction signal.
    if use_derived_features:
        # compare against baseline graph
        alt_graph = build_graph(train_part, False)
    else:
        # compare against derived-features graph
        alt_graph = build_graph(train_part, True)

    alt_learner = alt_graph.skb.make_learner(fitted=True)
    pred_2 = alt_learner.predict({"data": valid_part})
    pred_2 = np.clip(np.asarray(pred_2, dtype=float), holdout_min, holdout_max)

    delta = pred_2 - pred_1
    y_true = valid_part[target_col].to_numpy(dtype=float)

    alpha_grid = np.round(np.arange(-0.5, 1.5001, 0.05), 2)
    variant_best_score = np.inf
    variant_best_alpha = 0.0

    for alpha in alpha_grid:
        blended = pred_1 + alpha * delta
        blended = np.clip(blended, holdout_min, holdout_max)
        score = mean_squared_error(y_true, blended) ** 0.5
        if score < variant_best_score:
            variant_best_score = score
            variant_best_alpha = float(alpha)

    print(f"Ablation[{variant_name}] RMSE: {variant_best_score}")

    if variant_best_score < best_score:
        best_score = variant_best_score
        best_variant = variant_name
        best_alpha = variant_best_alpha
        best_holdout_min = holdout_min
        best_holdout_max = holdout_max

final_validation_score = best_score
print(f"Best ablation variant: {best_variant} | RMSE: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")

# Train both pipelines on full train_df for final test prediction
full_data = skrub.var("data", train_df)

def build_graph_full(dataframe, use_derived_features):
    data = skrub.var("data", dataframe)

    if use_derived_features:
        data = data.skb.apply_func(add_guarded_housing_ratios)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()
    X = X.skb.apply(ApplyToCols(SquashingScaler(max_absolute_value=3), cols=skrub.selectors.numeric()))
    vectorizer = skrub.TableVectorizer()
    pred_graph = X.skb.apply(vectorizer).skb.apply(model, y=y)
    return pred_graph

base_graph_full = build_graph_full(train_df, best_variant == "derived_features")
base_learner_full = base_graph_full.skb.make_learner(fitted=True)
test_pred_1 = base_learner_full.predict({"data": test_df})

alt_graph_full = build_graph_full(train_df, best_variant != "derived_features")
alt_learner_full = alt_graph_full.skb.make_learner(fitted=True)
test_pred_2 = alt_learner_full.predict({"data": test_df})

test_pred_1 = np.clip(np.asarray(test_pred_1, dtype=float), best_holdout_min, best_holdout_max)
test_pred_2 = np.clip(np.asarray(test_pred_2, dtype=float), best_holdout_min, best_holdout_max)

if abs(best_alpha) < 1e-6:
    test_pred = test_pred_1
else:
    test_pred = test_pred_1 + best_alpha * (test_pred_2 - test_pred_1)

test_pred = np.clip(test_pred, best_holdout_min, best_holdout_max)

submission = pd.DataFrame({
    "Id": test_df["Id"] if "Id" in test_df.columns else np.arange(len(test_df)),
    target_col: test_pred,
})
submission.to_csv("submission.csv", index=False)
