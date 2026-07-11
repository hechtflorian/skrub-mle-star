
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

random_state = 42
target_col = "Attrition"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def drop_constants_and_uninformative(df):
    out = df.copy()
    nunique = out.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        out = out.drop(columns=constant_cols)
    return out

# Shared DataOps root
data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(drop_constants_and_uninformative)
X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_train[target_col].skb.mark_as_y()

# Shared vectorizer
shared_vectorizer = skrub.TableVectorizer()

# Leg 1: CatBoost
cat_model = CatBoostClassifier(
    verbose=0,
    random_state=random_state,
    loss_function="Logloss",
    iterations=300,
    depth=6,
    learning_rate=0.08,
)

# Leg 2: LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph_cat = X.skb.apply(shared_vectorizer).skb.apply(cat_model, y=y)
pred_graph_lgbm = X.skb.apply(shared_vectorizer).skb.apply(lgbm_model, y=y)

learner_cat = pred_graph_cat.skb.make_learner(fitted=True)
learner_lgbm = pred_graph_lgbm.skb.make_learner(fitted=True)

def to_valid_pred(learner, df):
    try:
        pred = learner.predict({"data": df})
    except Exception:
        pred = learner.predict_proba({"data": df})
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        pred = pred[:, 1]
    return pred.astype(float).ravel()

valid_pred_cat = to_valid_pred(learner_cat, valid_part)
valid_pred_lgbm = to_valid_pred(learner_lgbm, valid_part)

# Raw probability blend candidates
raw_weights = [0.70, 0.80, 0.90, 0.50]
raw_scores = {}
for w_cat in raw_weights:
    raw_blend = w_cat * valid_pred_cat + (1.0 - w_cat) * valid_pred_lgbm
    raw_scores[w_cat] = roc_auc_score(valid_part[target_col], raw_blend)
    print(f"Ablation[raw_blend_cat_{w_cat:.2f}] roc_auc: {raw_scores[w_cat]}")

# Rank-normalized blend candidates
valid_pred_cat_rank = rankdata(valid_pred_cat, method="average") / len(valid_pred_cat)
valid_pred_lgbm_rank = rankdata(valid_pred_lgbm, method="average") / len(valid_pred_lgbm)

rank_scores = {}
rank_blends = {}
for w_cat in raw_weights:
    rank_blend = w_cat * valid_pred_cat_rank + (1.0 - w_cat) * valid_pred_lgbm_rank
    rank_blends[w_cat] = rank_blend
    rank_scores[w_cat] = roc_auc_score(valid_part[target_col], rank_blend)
    print(f"Ablation[rank_blend_cat_{w_cat:.2f}] roc_auc: {rank_scores[w_cat]}")

# Blend of blends: average raw and rank blend for each weight
bob_scores = {}
for w_cat in raw_weights:
    raw_blend = w_cat * valid_pred_cat + (1.0 - w_cat) * valid_pred_lgbm
    rank_blend = rank_blends[w_cat]
    blend_of_blends = 0.5 * raw_blend + 0.5 * rank_blend
    bob_scores[w_cat] = roc_auc_score(valid_part[target_col], blend_of_blends)
    print(f"Ablation[blend_of_blends_cat_{w_cat:.2f}] roc_auc: {bob_scores[w_cat]}")

# Select best by holdout AUC
best_raw_w = max(raw_scores, key=raw_scores.get)
best_rank_w = max(rank_scores, key=rank_scores.get)
best_bob_w = max(bob_scores, key=bob_scores.get)

best_candidates = {
    f"raw_{best_raw_w:.2f}": raw_scores[best_raw_w],
    f"rank_{best_rank_w:.2f}": rank_scores[best_rank_w],
    f"bob_{best_bob_w:.2f}": bob_scores[best_bob_w],
}

best_name = max(best_candidates, key=best_candidates.get)

if best_name.startswith("raw_"):
    final_validation_pred = best_raw_w * valid_pred_cat + (1.0 - best_raw_w) * valid_pred_lgbm
elif best_name.startswith("rank_"):
    final_validation_pred = best_rank_w * valid_pred_cat_rank + (1.0 - best_rank_w) * valid_pred_lgbm_rank
else:
    raw_blend = best_bob_w * valid_pred_cat + (1.0 - best_bob_w) * valid_pred_lgbm
    rank_blend = best_bob_w * valid_pred_cat_rank + (1.0 - best_bob_w) * valid_pred_lgbm_rank
    final_validation_pred = 0.5 * raw_blend + 0.5 * rank_blend

final_validation_score = roc_auc_score(valid_part[target_col], final_validation_pred)
print(f"Final Validation Performance: {final_validation_score}")
