
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
