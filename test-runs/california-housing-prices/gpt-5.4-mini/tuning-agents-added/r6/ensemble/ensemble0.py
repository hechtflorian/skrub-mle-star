
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
# Solution 1 pipeline
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

valid_pred_1 = np.asarray(val_learner_1.predict({"data": valid_part})).ravel()

# -------------------------
# Solution 2 pipeline
# -------------------------
# Kept structurally similar to the first pipeline, but with a conservative
# model variation to enable blending without changing the preprocessing flow.
data_train_2 = skrub.var("data", train_part)
data_train_fe_2 = data_train_2.skb.apply_func(add_house_feature_ratios)

X_train_2 = data_train_fe_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_2 = data_train_fe_2[target_col].skb.mark_as_y()

vectorizer_2 = skrub.TableVectorizer()

model_2 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.02,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)

pred_graph_2 = X_train_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train_2)
val_learner_2 = pred_graph_2.skb.make_learner(fitted=True)

valid_pred_2 = np.asarray(val_learner_2.predict({"data": valid_part})).ravel()

y_valid = valid_part[target_col].to_numpy()

def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5

def rank_blend(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return 0.5 * ra + 0.5 * rb

# Grid search for weighted average
best_w = None
best_weighted_rmse = np.inf
best_weighted_pred = None
for w in np.arange(0.1, 1.0, 0.1):
    pred = w * valid_pred_1 + (1.0 - w) * valid_pred_2
    score = rmse(y_valid, pred)
    if score < best_weighted_rmse:
        best_weighted_rmse = score
        best_w = float(w)
        best_weighted_pred = pred

# Rank-based candidate
valid_rank_pred = rank_blend(valid_pred_1, valid_pred_2)
rank_rmse = rmse(y_valid, valid_rank_pred)

# Decide final merge rule conservatively
single_rmse_1 = rmse(y_valid, valid_pred_1)
single_rmse_2 = rmse(y_valid, valid_pred_2)
best_single_rmse = min(single_rmse_1, single_rmse_2)

if min(best_weighted_rmse, rank_rmse) < best_single_rmse:
    if rank_rmse <= best_weighted_rmse:
        final_merge_rule = ("rank", None)
        final_validation_pred = valid_rank_pred
        final_validation_score = rank_rmse
    else:
        final_merge_rule = ("weighted", best_w)
        final_validation_pred = best_weighted_pred
        final_validation_score = best_weighted_rmse
else:
    if single_rmse_1 <= single_rmse_2:
        final_merge_rule = ("single", 1)
        final_validation_pred = valid_pred_1
        final_validation_score = single_rmse_1
    else:
        final_merge_rule = ("single", 2)
        final_validation_pred = valid_pred_2
        final_validation_score = single_rmse_2

print(f"Final Validation Performance: {final_validation_score}")

# -------------------------
# Refit both base solutions on full data, then blend test predictions
# -------------------------
data_full_1 = skrub.var("data", train_df)
data_full_fe_1 = data_full_1.skb.apply_func(add_house_feature_ratios)

X_full_1 = data_full_fe_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_1 = data_full_fe_1[target_col].skb.mark_as_y()

full_pred_graph_1 = X_full_1.skb.apply(vectorizer_1).skb.apply(model_1, y=y_full_1)
full_learner_1 = full_pred_graph_1.skb.make_learner(fitted=True)

test_pred_1 = np.asarray(full_learner_1.predict({"data": test_df})).ravel()

data_full_2 = skrub.var("data", train_df)
data_full_fe_2 = data_full_2.skb.apply_func(add_house_feature_ratios)

X_full_2 = data_full_fe_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_2 = data_full_fe_2[target_col].skb.mark_as_y()

full_pred_graph_2 = X_full_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_full_2)
full_learner_2 = full_pred_graph_2.skb.make_learner(fitted=True)

test_pred_2 = np.asarray(full_learner_2.predict({"data": test_df})).ravel()

if final_merge_rule[0] == "weighted":
    w = final_merge_rule[1]
    test_pred = w * test_pred_1 + (1.0 - w) * test_pred_2
elif final_merge_rule[0] == "rank":
    test_pred = rank_blend(test_pred_1, test_pred_2)
else:
    test_pred = test_pred_1 if final_merge_rule[1] == 1 else test_pred_2

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
