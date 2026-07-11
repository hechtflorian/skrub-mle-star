
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
TARGET_COL = "Attrition"
ID_COL = "id"

train_path = "./input/train.csv"

train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_df[TARGET_COL],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_train = data_train[TARGET_COL].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_X_train = X_train.skb.apply(vectorizer)
cat_model = CatBoostClassifier(
    random_state=RANDOM_STATE,
    verbose=0,
    cat_features=None,
)
cat_predictor = cat_X_train.skb.apply(cat_model, y=y_train)

lgbm_X_train = X_train.skb.apply(vectorizer)
lgbm_model = LGBMClassifier(
    random_state=RANDOM_STATE,
    verbose=-1,
)
lgbm_predictor = lgbm_X_train.skb.apply(lgbm_model, y=y_train)

cat_learner = cat_predictor.skb.make_learner(fitted=True)
lgbm_learner = lgbm_predictor.skb.make_learner(fitted=True)

cat_valid_pred = np.asarray(cat_learner.predict({"data": valid_part}))
lgbm_valid_pred = np.asarray(lgbm_learner.predict({"data": valid_part}))

if set(np.unique(cat_valid_pred)).issubset({0, 1}):
    cat_valid_pred = cat_learner.predict_proba({"data": valid_part})[:, 1]
if set(np.unique(lgbm_valid_pred)).issubset({0, 1}):
    lgbm_valid_pred = lgbm_learner.predict_proba({"data": valid_part})[:, 1]

cat_valid_pred = np.asarray(cat_valid_pred, dtype=float).ravel()
lgbm_valid_pred = np.asarray(lgbm_valid_pred, dtype=float).ravel()

def rank_normalize(x):
    s = pd.Series(np.asarray(x, dtype=float).ravel())
    ranks = s.rank(method="average", pct=True).to_numpy()
    return ranks.astype(float)

y_valid = valid_part[TARGET_COL].to_numpy()

cat_auc = roc_auc_score(y_valid, cat_valid_pred)
lgbm_auc = roc_auc_score(y_valid, lgbm_valid_pred)

raw_weight_candidates = [(0.50, 0.50)]
if cat_auc > lgbm_auc:
    raw_weight_candidates.extend([(0.55, 0.45), (0.60, 0.40)])
elif lgbm_auc > cat_auc:
    raw_weight_candidates.extend([(0.45, 0.55), (0.40, 0.60)])

alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

best_auc = -np.inf
best_pred = None

best_raw_auc = -np.inf
best_raw_pred = None

for w_cat, w_lgbm in raw_weight_candidates:
    raw_mean = w_cat * cat_valid_pred + w_lgbm * lgbm_valid_pred

    rank_cat = rank_normalize(cat_valid_pred)
    rank_lgbm = rank_normalize(lgbm_valid_pred)
    rank_mean = w_cat * rank_cat + w_lgbm * rank_lgbm
    consensus = 1.0 - np.abs(rank_cat - rank_lgbm)

    raw_auc = roc_auc_score(y_valid, raw_mean)
    if raw_auc > best_raw_auc:
        best_raw_auc = raw_auc
        best_raw_pred = raw_mean.copy()

    for alpha in alpha_grid:
        adaptive_blend = alpha * raw_mean + (1.0 - alpha) * (
            consensus * rank_mean + (1.0 - consensus) * raw_mean
        )
        auc = roc_auc_score(y_valid, adaptive_blend)
        if auc > best_auc:
            best_auc = auc
            best_pred = adaptive_blend.copy()

if best_auc > best_raw_auc:
    final_validation_score = best_auc
else:
    final_validation_score = best_raw_auc
    best_pred = best_raw_pred

print(f"Final Validation Performance: {final_validation_score}")

test_df = pd.read_csv("./input/test.csv")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_full = data_full[TARGET_COL].skb.mark_as_y()

cat_X_full = X_full.skb.apply(vectorizer)
cat_full_predictor = cat_X_full.skb.apply(cat_model, y=y_full)

lgbm_X_full = X_full.skb.apply(vectorizer)
lgbm_full_predictor = lgbm_X_full.skb.apply(lgbm_model, y=y_full)

cat_full_learner = cat_full_predictor.skb.make_learner(fitted=True)
lgbm_full_learner = lgbm_full_predictor.skb.make_learner(fitted=True)

cat_test_pred = np.asarray(cat_full_learner.predict({"data": test_df}))
lgbm_test_pred = np.asarray(lgbm_full_learner.predict({"data": test_df}))

if set(np.unique(cat_test_pred)).issubset({0, 1}):
    cat_test_pred = cat_full_learner.predict_proba({"data": test_df})[:, 1]
if set(np.unique(lgbm_test_pred)).issubset({0, 1}):
    lgbm_test_pred = lgbm_full_learner.predict_proba({"data": test_df})[:, 1]

cat_test_pred = np.asarray(cat_test_pred, dtype=float).ravel()
lgbm_test_pred = np.asarray(lgbm_test_pred, dtype=float).ravel()

test_rank_cat = rank_normalize(cat_test_pred)
test_rank_lgbm = rank_normalize(lgbm_test_pred)
test_consensus = 1.0 - np.abs(test_rank_cat - test_rank_lgbm)

best_test_pred = None
current_best_auc = -np.inf

for w_cat, w_lgbm in raw_weight_candidates:
    test_raw_mean = w_cat * cat_test_pred + w_lgbm * lgbm_test_pred
    test_rank_mean = w_cat * test_rank_cat + w_lgbm * test_rank_lgbm

    if best_auc > best_raw_auc:
        for alpha in alpha_grid:
            test_adaptive_blend = alpha * test_raw_mean + (1.0 - alpha) * (
                test_consensus * test_rank_mean + (1.0 - test_consensus) * test_raw_mean
            )
            validation_like_score = roc_auc_score(y_valid, alpha * (w_cat * cat_valid_pred + w_lgbm * lgbm_valid_pred) + (1.0 - alpha) * (
                (1.0 - np.abs(rank_normalize(cat_valid_pred) - rank_normalize(lgbm_valid_pred))) * (w_cat * rank_normalize(cat_valid_pred) + w_lgbm * rank_normalize(lgbm_valid_pred))
                + (1.0 - (1.0 - np.abs(rank_normalize(cat_valid_pred) - rank_normalize(lgbm_valid_pred)))) * (w_cat * cat_valid_pred + w_lgbm * lgbm_valid_pred)
            ))
            if validation_like_score > current_best_auc:
                current_best_auc = validation_like_score
                best_test_pred = test_adaptive_blend.copy()
    else:
        validation_like_score = roc_auc_score(y_valid, w_cat * cat_valid_pred + w_lgbm * lgbm_valid_pred)
        if validation_like_score > current_best_auc:
            current_best_auc = validation_like_score
            best_test_pred = test_raw_mean.copy()

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame(
    {
        "EmployeeNumber": test_df[ID_COL],
        TARGET_COL: np.asarray(best_test_pred, dtype=float).ravel(),
    }
)
submission.to_csv("./final/submission.csv", index=False)
