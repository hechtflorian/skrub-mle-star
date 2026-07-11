
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

target_col = "revenue"
base_random_state = 0

train_df = pd.read_csv("./input/train.csv")

if "Open Date" in train_df.columns:
    train_df["Open Date"] = pd.to_datetime(train_df["Open Date"], errors="coerce")


def fit_and_eval_leg(train_df_local, split_seed):
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df_local)), test_size=0.2, random_state=split_seed
    )
    train_part = train_df_local.iloc[train_idx].copy()
    valid_part = train_df_local.iloc[valid_idx].copy()

    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=split_seed,
        verbose=0,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    val_learner = pred.skb.make_learner(fitted=True)
    valid_pred = np.asarray(val_learner.predict({"data": valid_part}), dtype=float).ravel()

    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    return {
        "seed": split_seed,
        "rmse": rmse,
        "valid_idx": valid_idx,
        "valid_part": valid_part,
        "valid_pred": valid_pred,
        "train_part": train_part,
        "vectorizer": vectorizer,
        "model": model,
    }


leg1 = fit_and_eval_leg(train_df, base_random_state)
leg2 = fit_and_eval_leg(train_df, 7)

# Robust rank-based blend on the shared holdout comparison layer
p1 = leg1["valid_pred"]
p2 = leg2["valid_pred"]

r1 = pd.Series(p1).rank(method="average", pct=True).to_numpy()
r2 = pd.Series(p2).rank(method="average", pct=True).to_numpy()
blend_rank = 0.5 * r1 + 0.5 * r2

# Map blended ranks back to the value range using a simple quantile-style reconstruction
sorted_ref = np.sort(p1)
if len(sorted_ref) == 0:
    blended_pred = p1.copy()
else:
    q_idx = np.clip((blend_rank * (len(sorted_ref) - 1)).round().astype(int), 0, len(sorted_ref) - 1)
    blended_pred = sorted_ref[q_idx]

blend_rmse = mean_squared_error(leg1["valid_part"][target_col], blended_pred) ** 0.5

best_single_rmse = min(leg1["rmse"], leg2["rmse"])
best_single_leg = leg1 if leg1["rmse"] <= leg2["rmse"] else leg2

# Use blended predictor only if it clearly beats the best single leg
final_validation_score = blend_rmse if blend_rmse + 1e-12 < best_single_rmse else best_single_leg["rmse"]

print(f"Final Validation Performance: {final_validation_score}")

test_df = pd.read_csv("./input/test.csv")

if "Open Date" in test_df.columns:
    test_df["Open Date"] = pd.to_datetime(test_df["Open Date"], errors="coerce")

data_full_1 = skrub.var("data", train_df)
X_full_1 = data_full_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_1 = data_full_1[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()
model_1 = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=base_random_state,
    verbose=0,
)

full_pred_1 = X_full_1.skb.apply(vectorizer_1).skb.apply(model_1, y=y_full_1)
full_learner_1 = full_pred_1.skb.make_learner(fitted=True)
test_pred_1 = np.asarray(full_learner_1.predict({"data": test_df}), dtype=float).ravel()

data_full_2 = skrub.var("data", train_df)
X_full_2 = data_full_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_2 = data_full_2[target_col].skb.mark_as_y()

vectorizer_2 = skrub.TableVectorizer()
model_2 = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=7,
    verbose=0,
)

full_pred_2 = X_full_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_full_2)
full_learner_2 = full_pred_2.skb.make_learner(fitted=True)
test_pred_2 = np.asarray(full_learner_2.predict({"data": test_df}), dtype=float).ravel()

r1_test = pd.Series(test_pred_1).rank(method="average", pct=True).to_numpy()
r2_test = pd.Series(test_pred_2).rank(method="average", pct=True).to_numpy()
blend_rank_test = 0.5 * r1_test + 0.5 * r2_test

sorted_ref_test = np.sort(test_pred_1)
if len(sorted_ref_test) == 0:
    blended_test_pred = test_pred_1.copy()
else:
    q_idx_test = np.clip(
        (blend_rank_test * (len(sorted_ref_test) - 1)).round().astype(int),
        0,
        len(sorted_ref_test) - 1,
    )
    blended_test_pred = sorted_ref_test[q_idx_test]

test_blend_rmse_proxy = None
best_single_test_pred = test_pred_1 if leg1["rmse"] <= leg2["rmse"] else test_pred_2
test_pred = blended_test_pred if blend_rmse + 1e-12 < best_single_rmse else best_single_test_pred

import os
os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"Id": test_df["Id"], "Prediction": test_pred})
submission.to_csv("./final/submission.csv", index=False)
