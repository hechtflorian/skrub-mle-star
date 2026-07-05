import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred_chain = build_graph(data_train)
    learner = pred_chain.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred)
    if valid_pred.dtype != bool:
        unique_vals = set(np.unique(valid_pred).tolist())
        if unique_vals <= {0, 1}:
            valid_pred = valid_pred.astype(int).astype(bool)
        else:
            valid_pred = pd.Series(valid_pred).astype(str).isin(["True", "true", "1"]).to_numpy()
    score = accuracy_score(valid_part[target_col].values, valid_pred)
    print(f"Ablation[{variant_name}] accuracy_score: {score}")
    return score


def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def no_passengerid_graph(data_train):
    X_train = data_train.drop(columns=[target_col, "PassengerId"], errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def high_cardinality_drop_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer(high_cardinality="drop")
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def cleaner_then_vectorizer_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    cleaner = skrub.Cleaner(drop_if_constant=True)
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
    )
    return X_train.skb.apply(cleaner).skb.apply(vectorizer).skb.apply(model, y=y_train)


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["drop_passengerid"] = score_variant("drop_passengerid", no_passengerid_graph)
scores["drop_high_cardinality"] = score_variant("drop_high_cardinality", high_cardinality_drop_graph)
scores["cleaner_before_vectorizer"] = score_variant("cleaner_before_vectorizer", cleaner_then_vectorizer_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy_score: {best_score}")