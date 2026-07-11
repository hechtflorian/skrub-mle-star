
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

target_col = "Transported"
random_state = 42
metric_label = "accuracy"

train_df = pd.read_csv(TRAIN_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def to_bool_series(values, index=None):
    s = pd.Series(values, index=index)
    mapped = s.astype(str).map({"True": True, "False": False})
    return mapped.where(mapped.notna(), s).astype(bool)


def make_lgbm():
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )


def make_cat():
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
    )


def fe_drop_id_like(df):
    out = df.copy()
    drop_cols = [c for c in ["PassengerId", "Name"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def fe_split_cabin(df):
    out = df.copy()
    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].fillna("Missing/Missing/Missing").astype(str).str.split("/", expand=True)
        if cabin_parts.shape[1] >= 3:
            out["CabinDeck"] = cabin_parts[0]
            out["CabinNum"] = cabin_parts[1]
            out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"])
    return out


def build_lgbm_graph(data_train, fe_func=None):
    if fe_func is not None:
        data_train = data_train.skb.apply_func(fe_func)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(make_lgbm(), y=y_train)


def build_cat_graph(data_train, fe_func=None):
    if fe_func is not None:
        data_train = data_train.skb.apply_func(fe_func)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(make_cat(), y=y_train)


def score_variant(variant_name, lgbm_fe=None, cat_fe=None, use_lgbm=True, use_cat=True):
    valid_true = to_bool_series(valid_part[target_col], index=valid_part.index)
    preds_num = []

    if use_lgbm:
        data_train_lgbm = skrub.var("data", train_part)
        pred_lgbm = build_lgbm_graph(data_train_lgbm, fe_func=lgbm_fe)
        learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
        valid_pred_lgbm = learner_lgbm.predict({"data": valid_part})
        preds_num.append(to_bool_series(valid_pred_lgbm, index=valid_part.index).astype(int))

    if use_cat:
        data_train_cat = skrub.var("data", train_part)
        pred_cat = build_cat_graph(data_train_cat, fe_func=cat_fe)
        learner_cat = pred_cat.skb.make_learner(fitted=True)
        valid_pred_cat = learner_cat.predict({"data": valid_part})
        preds_num.append(to_bool_series(valid_pred_cat, index=valid_part.index).astype(int))

    if len(preds_num) == 2:
        ensemble_vote = ((preds_num[0] + preds_num[1]) >= 1).astype(bool)
    else:
        ensemble_vote = preds_num[0].astype(bool)

    score = accuracy_score(valid_true, ensemble_vote)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


scores = {}
scores["baseline_ensemble"] = score_variant("baseline_ensemble")
scores["drop_id_like"] = score_variant("drop_id_like", lgbm_fe=fe_drop_id_like, cat_fe=fe_drop_id_like)
scores["split_cabin"] = score_variant("split_cabin", lgbm_fe=fe_split_cabin, cat_fe=fe_split_cabin)
scores["lgbm_only"] = score_variant("lgbm_only", use_lgbm=True, use_cat=False)
scores["cat_only"] = score_variant("cat_only", use_lgbm=False, use_cat=True)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Most impactful ablation: {best_variant} | {metric_label}: {best_score}")
