
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")

target_col = "Transported"
metric_fn = accuracy_score
metric_label = "accuracy"

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = metric_fn(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

def no_cabin_graph(data_train):
    df = data_train.drop(columns=["Cabin"], errors="ignore")
    X_train = df.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = df[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

def no_name_graph(data_train):
    df = data_train.drop(columns=["Name"], errors="ignore")
    X_train = df.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = df[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["no_cabin"] = score_variant("no_cabin", no_cabin_graph)
scores["no_name"] = score_variant("no_name", no_name_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
print(f"Final Validation Performance: {best_score}")
