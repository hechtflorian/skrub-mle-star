import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

# Honest holdout split (same as original solution)
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred_chain = build_graph(data_train)
    learner = pred_chain.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred).ravel()
    valid_pred_bool = valid_pred > 0.5
    score = accuracy_score(valid_part[target_col].values, valid_pred_bool)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score


def make_baseline_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
        y=y,
    )


def fe_add_cabin_parts(df):
    out = df.copy()
    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype("string")
        parts = cabin.str.split("/", n=2, expand=True)
        if parts.shape[1] >= 1:
            out["CabinDeck"] = parts[0]
        if parts.shape[1] >= 2:
            out["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
        if parts.shape[1] >= 3:
            out["CabinSide"] = parts[2]
    return out


def make_fe_graph(data_train):
    data_fe = data_train.skb.apply_func(fe_add_cabin_parts)
    X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_fe[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
        y=y,
    )


def make_drop_passengerid_graph(data_train):
    X = data_train.drop(columns=[target_col, "PassengerId"], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
        y=y,
    )


def make_drop_cabin_graph(data_train):
    X = data_train.drop(columns=[target_col, "Cabin"], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
        y=y,
    )


scores = {}
scores["baseline"] = score_variant("baseline", make_baseline_graph)
scores["add_cabin_parts"] = score_variant("add_cabin_parts", make_fe_graph)
scores["drop_passengerid"] = score_variant("drop_passengerid", make_drop_passengerid_graph)
scores["drop_cabin"] = score_variant("drop_cabin", make_drop_cabin_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {best_score}")