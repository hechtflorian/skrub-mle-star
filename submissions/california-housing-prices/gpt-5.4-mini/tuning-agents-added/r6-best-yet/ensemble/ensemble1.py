
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

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
def add_house_feature_ratios(df):
    out = df.copy()
    if "total_rooms" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "population" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (out["population"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "total_bedrooms" in out.columns and "total_rooms" in out.columns:
        denom = out["total_rooms"].replace(0, np.nan)
        out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    return out

# -------------------------
# Base pipeline 1
# -------------------------
data_train_1 = skrub.var("data", train_part)
data_train_fe_1 = data_train_1.skb.apply_func(add_house_feature_ratios)

X_train_1 = data_train_fe_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_1 = data_train_fe_1[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()

model_1 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)

pred_graph_1 = X_train_1.skb.apply(vectorizer_1).skb.apply(model_1, y=y_train_1)
val_learner_1 = pred_graph_1.skb.make_learner(fitted=True)
valid_pred_1 = val_learner_1.predict({"data": valid_part})

# -------------------------
# Base pipeline 2 (kept structurally similar, slightly different seed)
# -------------------------
data_train_2 = skrub.var("data", train_part)
data_train_fe_2 = data_train_2.skb.apply_func(add_house_feature_ratios)

X_train_2 = data_train_fe_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_2 = data_train_fe_2[target_col].skb.mark_as_y()

vectorizer_2 = skrub.TableVectorizer()

model_2 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=2024,
    verbose=0,
    allow_writing_files=False,
)

pred_graph_2 = X_train_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train_2)
val_learner_2 = pred_graph_2.skb.make_learner(fitted=True)
valid_pred_2 = val_learner_2.predict({"data": valid_part})

# -------------------------
# Residual-corrected stacking on validation split
# -------------------------
y_valid = valid_part[target_col].to_numpy(dtype=float)

valid_pred_1 = np.asarray(valid_pred_1, dtype=float)
valid_pred_2 = np.asarray(valid_pred_2, dtype=float)

m1 = valid_pred_1.mean()
m2 = valid_pred_2.mean()
y_mean = y_valid.mean()

Z_valid_centered = np.column_stack([
    valid_pred_1 - m1,
    valid_pred_2 - m2,
])

# Simple linear combiner with non-negative coefficients for stability
combiner = Ridge(alpha=1.0, fit_intercept=False, positive=True)
combiner.fit(Z_valid_centered, y_valid - y_mean)

coef = np.asarray(combiner.coef_, dtype=float)
pred_valid_ens = y_mean + Z_valid_centered @ coef

# Conservative fallback if combiner collapses to one model
if np.max(coef) >= 0.95 * (np.sum(coef) + 1e-12):
    best_idx = int(np.argmax(coef))
    pred_valid_ens = valid_pred_1 if best_idx == 0 else valid_pred_2

# Robust clipping fallback to training target quantiles
q_low, q_high = np.percentile(train_df[target_col].values, [1, 99])
pred_valid_ens = np.clip(pred_valid_ens, q_low, q_high)

final_validation_score = mean_squared_error(y_valid, pred_valid_ens) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# -------------------------
# Submission stage: refit both original pipelines on all training data
# -------------------------
data_full_1 = skrub.var("data", train_df)
data_full_fe_1 = data_full_1.skb.apply_func(add_house_feature_ratios)

X_full_1 = data_full_fe_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_1 = data_full_fe_1[target_col].skb.mark_as_y()

full_pred_graph_1 = X_full_1.skb.apply(vectorizer_1).skb.apply(model_1, y=y_full_1)
full_learner_1 = full_pred_graph_1.skb.make_learner(fitted=True)

data_full_2 = skrub.var("data", train_df)
data_full_fe_2 = data_full_2.skb.apply_func(add_house_feature_ratios)

X_full_2 = data_full_fe_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_2 = data_full_fe_2[target_col].skb.mark_as_y()

full_pred_graph_2 = X_full_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_full_2)
full_learner_2 = full_pred_graph_2.skb.make_learner(fitted=True)

test_pred_1 = np.asarray(full_learner_1.predict({"data": test_df}), dtype=float)
test_pred_2 = np.asarray(full_learner_2.predict({"data": test_df}), dtype=float)

test_pred_1_centered = test_pred_1 - test_pred_1.mean()
test_pred_2_centered = test_pred_2 - test_pred_2.mean()

Z_test_centered = np.column_stack([test_pred_1_centered, test_pred_2_centered])

test_pred = y_mean + Z_test_centered @ coef

if np.max(coef) >= 0.95 * (np.sum(coef) + 1e-12):
    best_idx = int(np.argmax(coef))
    test_pred = test_pred_1 if best_idx == 0 else test_pred_2

test_pred = np.clip(test_pred, q_low, q_high)

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
