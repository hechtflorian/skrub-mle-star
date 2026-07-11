import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "Attrition"
metric_label = "ROC-AUC"

input_dir = Path("./input")
train_path = input_dir / "train.csv"
train_df = pd.read_csv(train_path)

train_part, valid_part = train_test_split(
    train_df,
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)

def build_lgbm():
    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=random_state,
        verbose=-1,
    )

def build_cat():
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5,
        random_seed=random_state,
        verbose=0,
    )

def drop_redundant_columns(df):
    out = df.copy()
    cols_to_drop = [c for c in ["id", "EmployeeCount", "Over18", "StandardHours"] if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop, errors="ignore")
    return out

def score_variant(variant_name, use_lgbm=True, use_cat=True, drop_redundant=False):
    data_train = skrub.var("data", train_part)

    if drop_redundant:
        data_train = data_train.skb.apply_func(drop_redundant_columns)

    X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    preds = []

    if use_lgbm:
        vectorizer_lgbm = skrub.TableVectorizer()
        lgbm_predictor = X_train.skb.apply(vectorizer_lgbm).skb.apply(build_lgbm(), y=y_train)
        lgbm_learner = lgbm_predictor.skb.make_learner(fitted=True)
        pred_lgbm = np.asarray(lgbm_learner.predict({"data": valid_part}), dtype=float).ravel()
        preds.append(pred_lgbm)

    if use_cat:
        vectorizer_cat = skrub.TableVectorizer()
        cat_predictor = X_train.skb.apply(vectorizer_cat).skb.apply(build_cat(), y=y_train)
        cat_learner = cat_predictor.skb.make_learner(fitted=True)
        pred_cat = np.asarray(cat_learner.predict({"data": valid_part}), dtype=float).ravel()
        preds.append(pred_cat)

    valid_pred = np.mean(preds, axis=0)
    score = roc_auc_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

scores = {}
scores["baseline_ensemble"] = score_variant("baseline_ensemble", use_lgbm=True, use_cat=True, drop_redundant=False)
scores["lgbm_only"] = score_variant("lgbm_only", use_lgbm=True, use_cat=False, drop_redundant=False)
scores["catboost_only"] = score_variant("catboost_only", use_lgbm=False, use_cat=True, drop_redundant=False)
scores["drop_id_constant_cols"] = score_variant("drop_id_constant_cols", use_lgbm=True, use_cat=True, drop_redundant=True)

baseline_score = scores["baseline_ensemble"]
biggest_impact_name = None
biggest_impact_delta = -1.0

for name, score in scores.items():
    if name == "baseline_ensemble":
        continue
    delta = abs(score - baseline_score)
    if delta > biggest_impact_delta:
        biggest_impact_delta = delta
        biggest_impact_name = name

print(f"Part with largest impact vs baseline: {biggest_impact_name} | delta_{metric_label}: {biggest_impact_delta}")