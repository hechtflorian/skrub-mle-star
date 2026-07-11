
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


def get_positive_class_prob(learner, df):
    try:
        pred_proba = np.asarray(learner.predict_proba({"data": df}))
        if pred_proba.ndim == 2 and pred_proba.shape[1] > 1:
            return pred_proba[:, 1].astype(float)
        return pred_proba.ravel().astype(float)
    except Exception:
        pred = np.asarray(learner.predict({"data": df})).ravel()
        return pred.astype(float)


def rank_normalize(pred):
    pred = np.asarray(pred).ravel()
    return pd.Series(pred).rank(method="average", pct=True).to_numpy(dtype=float)


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

cat_valid_pred = get_positive_class_prob(cat_learner, valid_part)
lgbm_valid_pred = get_positive_class_prob(lgbm_learner, valid_part)

cat_auc = roc_auc_score(valid_part[TARGET_COL], cat_valid_pred)
lgbm_auc = roc_auc_score(valid_part[TARGET_COL], lgbm_valid_pred)

cat_rank = rank_normalize(cat_valid_pred)
lgbm_rank = rank_normalize(lgbm_valid_pred)
rank_mean_prob = 0.5 * cat_rank + 0.5 * lgbm_rank

weight_grid = [
    (0.5, 0.5, 0.0),
    (0.6, 0.4, 0.0),
    (0.4, 0.6, 0.0),
    (0.4, 0.4, 0.2),
    (0.35, 0.35, 0.30),
    (0.45, 0.45, 0.10),
]

best_score = -np.inf
best_pred = None
best_weights = None

for a, b, c in weight_grid:
    blend_pred = a * cat_valid_pred + b * lgbm_valid_pred + c * rank_mean_prob
    score = roc_auc_score(valid_part[TARGET_COL], blend_pred)
    if score > best_score:
        best_score = score
        best_pred = blend_pred
        best_weights = (a, b, c)

baseline_score = best_score

disagreement = np.abs(cat_valid_pred - lgbm_valid_pred)
cutoff = np.percentile(disagreement, 80)

if cat_auc >= lgbm_auc:
    low_disagree_pred = 0.5 * cat_valid_pred + 0.5 * lgbm_valid_pred
    high_disagree_pred = 0.65 * cat_valid_pred + 0.35 * lgbm_valid_pred
else:
    low_disagree_pred = 0.5 * cat_valid_pred + 0.5 * lgbm_valid_pred
    high_disagree_pred = 0.35 * cat_valid_pred + 0.65 * lgbm_valid_pred

disagreement_aware_pred = np.where(
    disagreement >= cutoff,
    high_disagree_pred,
    low_disagree_pred,
)

disagreement_aware_score = roc_auc_score(valid_part[TARGET_COL], disagreement_aware_pred)

if disagreement_aware_score > baseline_score:
    final_validation_score = disagreement_aware_score
else:
    final_validation_score = best_score

print(f"Final Validation Performance: {final_validation_score}")
