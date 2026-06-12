
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

for variant_name, use_derived_features in variants.items():
    pred_graph = build_graph(train_part, use_derived_features)
    val_learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})
    score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {score}")

    if score < best_score:
        best_score = score
        best_variant = variant_name

final_validation_score = best_score
print(f"Best ablation variant: {best_variant} | RMSE: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
