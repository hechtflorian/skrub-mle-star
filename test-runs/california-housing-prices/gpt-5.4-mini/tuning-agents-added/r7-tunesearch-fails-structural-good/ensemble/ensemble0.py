
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
def add_ratio_features(df):
    out = df.copy()
    if "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        if "total_rooms" in out.columns:
            out["rooms_per_household"] = (
                out["total_rooms"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "population" in out.columns:
            out["people_per_household"] = (
                out["population"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_ratio_features)

X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = CatBoostRegressor(
    iterations=5000,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200,
    allow_writing_files=False,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)

valid_catboost_pred = np.asarray(val_learner.predict({"data": valid_part}))

# Simple fallback baseline: target mean from the training split
fallback_value = float(train_part[target_col].mean())
valid_fallback_pred = np.full(shape=len(valid_part), fill_value=fallback_value, dtype=float)

# Lightweight convex blend at prediction layer only
blend_weight_catboost = 0.9
valid_pred = (
    blend_weight_catboost * valid_catboost_pred
    + (1.0 - blend_weight_catboost) * valid_fallback_pred
)

# Clip to plausible range for this regression target
valid_pred = np.clip(valid_pred, 0.0, 500000.0)

final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
