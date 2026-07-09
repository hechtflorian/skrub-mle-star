import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Personality"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

metric_fn = accuracy_score
metric_label = "accuracy"


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = metric_fn(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def baseline_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        verbose=0,
        random_seed=random_state,
    )
    return X.skb.apply(vectorizer).skb.apply(model, y=y)


def no_id_graph(data_train):
    df = data_train.skb.apply_func(
        lambda df: df.drop(columns=["id"], errors="ignore")
    )
    X = df.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = df[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        verbose=0,
        random_seed=random_state,
    )
    return X.skb.apply(vectorizer).skb.apply(model, y=y)


def high_card_drop_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer(high_cardinality="drop")
    model = CatBoostClassifier(
        loss_function="Logloss",
        verbose=0,
        random_seed=random_state,
    )
    return X.skb.apply(vectorizer).skb.apply(model, y=y)


def no_missing_cols_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        verbose=0,
        random_seed=random_state,
    )
    return X.skb.apply(vectorizer).skb.apply(model, y=y)


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["drop_id"] = score_variant("drop_id", no_id_graph)
scores["drop_high_cardinality"] = score_variant(
    "drop_high_cardinality", high_card_drop_graph
)
scores["baseline_again"] = score_variant("baseline_again", no_missing_cols_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")