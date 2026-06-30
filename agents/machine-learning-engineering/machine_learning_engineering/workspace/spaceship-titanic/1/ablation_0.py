
import os
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"

# Keep a small ablation-style validation split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def build_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    # Baseline backbone from the input solution family
    model = CatBoostClassifier(
        loss_function="Logloss",
        random_seed=42,
        verbose=0,
        iterations=300,
        depth=6,
        learning_rate=0.05,
    )

    encoder = skrub.TableVectorizer()
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

def score_variant(variant_name, data_part):
    data_train = skrub.var("data", data_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

scores = {}

# Baseline variant
scores["baseline"] = score_variant("baseline", train_part)

# Variant 1: slightly stronger regularization / simpler tree
def build_graph_variant_1(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    model = CatBoostClassifier(
        loss_function="Logloss",
        random_seed=42,
        verbose=0,
        iterations=250,
        depth=5,
        learning_rate=0.05,
    )
    encoder = skrub.TableVectorizer()
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

def score_variant_1(variant_name, data_part):
    data_train = skrub.var("data", data_part)
    pred = build_graph_variant_1(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

scores["compact_tree"] = score_variant_1("compact_tree", train_part)

# Variant 2: more trees, shallower depth
def build_graph_variant_2(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    model = CatBoostClassifier(
        loss_function="Logloss",
        random_seed=42,
        verbose=0,
        iterations=500,
        depth=4,
        learning_rate=0.03,
    )
    encoder = skrub.TableVectorizer()
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

def score_variant_2(variant_name, data_part):
    data_train = skrub.var("data", data_part)
    pred = build_graph_variant_2(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

scores["deeper_boost"] = score_variant_2("deeper_boost", train_part)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {best_score}")

# Final validation performance print required by the parser
final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")
