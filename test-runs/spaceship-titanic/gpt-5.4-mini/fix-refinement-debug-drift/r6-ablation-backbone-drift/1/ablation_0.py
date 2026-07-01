
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

def preprocess(df):
    out = df.copy()
    out["CabinDeck"] = out["Cabin"].astype("string").str.split("/").str[0]
    out["CabinNum"] = pd.to_numeric(out["Cabin"].astype("string").str.split("/").str[1], errors="coerce")
    out["CabinSide"] = out["Cabin"].astype("string").str.split("/").str[2]
    out["NameLength"] = out["Name"].astype("string").str.len()
    out["GroupId"] = out["PassengerId"].astype("string").str.split("_").str[0]
    out["GroupNum"] = pd.to_numeric(out["PassengerId"].astype("string").str.split("_").str[1], errors="coerce")
    out = out.drop(columns=["Cabin", "Name"], errors="ignore")
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

metric_label = "accuracy_score"

def score_variant(variant_name, model):
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    X_train = X_train.skb.apply_func(preprocess)
    vectorizer = skrub.TableVectorizer()
    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred).astype(bool)
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

scores = {}
scores["baseline_rf"] = score_variant(
    "baseline_rf",
    RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
)
scores["extra_trees"] = score_variant(
    "extra_trees",
    ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1),
)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")

final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")
