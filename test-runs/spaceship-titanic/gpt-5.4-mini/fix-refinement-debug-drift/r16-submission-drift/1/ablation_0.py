
import os
import re
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
test_size = 0.2
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

# Basic feature cleanup consistent across variants
def preprocess_df(df):
    out = df.copy()
    if "PassengerId" in out.columns:
        out["PassengerGroup"] = out["PassengerId"].astype(str).str.split("_").str[0]
        out["PassengerNum"] = out["PassengerId"].astype(str).str.split("_").str[1]
    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype(str).str.split("/", expand=True)
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNum"] = cabin_parts[1]
        out["CabinSide"] = cabin_parts[2]
    if "Name" in out.columns:
        out["NameLength"] = out["Name"].astype(str).str.len()
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

metric_label = "accuracy_score"

def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

def build_baseline(data_train):
    df = data_train.skb.apply_func(preprocess_df)
    X = df.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = df[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def build_no_name_feature(data_train):
    df = data_train.skb.apply_func(preprocess_df)
    df = df.drop(columns=["Name"], errors="ignore")
    X = df.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = df[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def build_no_cabin_feature(data_train):
    df = data_train.skb.apply_func(preprocess_df)
    df = df.drop(columns=["Cabin", "CabinDeck", "CabinNum", "CabinSide"], errors="ignore")
    X = df.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = df[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

scores = {}
scores["baseline"] = score_variant("baseline", build_baseline)
scores["no_name_feature"] = score_variant("no_name_feature", build_no_name_feature)
scores["no_cabin_feature"] = score_variant("no_cabin_feature", build_no_cabin_feature)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]

print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
print(f"Final Validation Performance: {best_score}")
