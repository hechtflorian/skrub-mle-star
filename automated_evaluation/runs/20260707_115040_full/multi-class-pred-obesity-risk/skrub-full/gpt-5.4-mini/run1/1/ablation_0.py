
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

random_state = 42
test_size = 0.2
target_col = "NObeyesdad"

train_df = pd.read_csv("./input/train.csv")
valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)[1]
train_idx = np.setdiff1d(np.arange(len(train_df)), valid_idx)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

def build_catboost_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=8,
        loss_function="MultiClass",
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def build_lgbm_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def build_catboost_no_id_graph(data_train):
    X = data_train.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=8,
        loss_function="MultiClass",
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

scores = {}
scores["catboost"] = score_variant("catboost", build_catboost_graph)
scores["lgbm"] = score_variant("lgbm", build_lgbm_graph)
scores["catboost_no_id"] = score_variant("catboost_no_id", build_catboost_no_id_graph)

best_variant = max(scores, key=scores.get)
final_validation_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
