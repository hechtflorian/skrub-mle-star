
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "yield"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)

drop_redundant_cols = ["id", "AverageOfUpperTRange", "AverageOfLowerTRange", "AverageRainingDays"]

X_train = (
    data_train
    .drop(columns=target_col, errors="ignore")
    .drop(columns=drop_redundant_cols, errors="ignore")
    .skb.mark_as_X()
)
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgb = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="mae",
    random_state=random_state,
    verbose=-1,
)

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=random_state,
    verbose=0,
)

# Keep Pattern A exactly: separate skrub DataOps learners per leg
lgb_predictor = X_train.skb.apply(vectorizer_lgb).skb.apply(lgb_model, y=y_train)
cat_predictor = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

lgb_learner = lgb_predictor.skb.make_learner(fitted=True)
cat_learner = cat_predictor.skb.make_learner(fitted=True)

valid_pred_lgb = np.asarray(lgb_learner.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}), dtype=float).ravel()

# Anchor blend
anchor_pred = 0.5 * valid_pred_lgb + 0.5 * valid_pred_cat
anchor_score = mean_absolute_error(valid_part[target_col], anchor_pred)

# Agreement-aware gated blend search on holdout
diff = np.abs(valid_pred_lgb - valid_pred_cat)

threshold_quantiles = [0.50, 0.65, 0.80, 0.90]
thresholds = np.quantile(diff, threshold_quantiles)
thresholds = np.unique(np.asarray(thresholds, dtype=float))

w_high_values = [1.0, 0.8, 0.2, 0.0]

best_score = anchor_score
best_pred = anchor_pred.copy()
min_improvement = 1e-6
best_threshold = None
best_w_high = None

for t in thresholds:
    for w_high in w_high_values:
        high_disagreement_pred = w_high * valid_pred_lgb + (1.0 - w_high) * valid_pred_cat
        candidate_pred = np.where(diff <= t, anchor_pred, high_disagreement_pred)
        candidate_score = mean_absolute_error(valid_part[target_col], candidate_pred)

        if candidate_score < best_score - min_improvement:
            best_score = candidate_score
            best_pred = candidate_pred
            best_threshold = float(t)
            best_w_high = float(w_high)

final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")

import os

test_df = pd.read_csv("./input/test.csv")

data_full = skrub.var("data", train_df)

X_full = (
    data_full
    .drop(columns=target_col, errors="ignore")
    .drop(columns=drop_redundant_cols, errors="ignore")
    .skb.mark_as_X()
)
y_full = data_full[target_col].skb.mark_as_y()

full_lgb_predictor = X_full.skb.apply(vectorizer_lgb).skb.apply(lgb_model, y=y_full)
full_cat_predictor = X_full.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_full)

full_lgb_learner = full_lgb_predictor.skb.make_learner(fitted=True)
full_cat_learner = full_cat_predictor.skb.make_learner(fitted=True)

test_pred_lgb = np.asarray(full_lgb_learner.predict({"data": test_df}), dtype=float).ravel()
test_pred_cat = np.asarray(full_cat_learner.predict({"data": test_df}), dtype=float).ravel()

test_anchor_pred = 0.5 * test_pred_lgb + 0.5 * test_pred_cat

if best_threshold is None:
    test_pred = test_anchor_pred
else:
    test_diff = np.abs(test_pred_lgb - test_pred_cat)
    test_high_disagreement_pred = best_w_high * test_pred_lgb + (1.0 - best_w_high) * test_pred_cat
    test_pred = np.where(test_diff <= best_threshold, test_anchor_pred, test_high_disagreement_pred)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({
    "id": test_df["id"],
    "yield": test_pred,
})
submission.to_csv("./final/submission.csv", index=False)
