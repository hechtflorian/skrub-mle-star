
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

random_state = 42
target_col = "NObeyesdad"
metric_label = "accuracy"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

cat_features = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def baseline_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = LGBMClassifier(
        random_state=random_state,
        verbose=-1,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
    )
    return X.skb.apply(vectorizer).skb.apply(model, y=y)


def cat_with_tablevectorizer_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="MultiClass",
        random_state=random_state,
        verbose=0,
    )
    return X.skb.apply(vectorizer).skb.apply(model, y=y)


def cat_no_tablevectorizer_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    model = CatBoostClassifier(
        cat_features=cat_features,
        loss_function="MultiClass",
        random_state=random_state,
        verbose=0,
    )
    return X.skb.apply(model, y=y)


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["cat_with_tablevectorizer"] = score_variant(
    "cat_with_tablevectorizer", cat_with_tablevectorizer_graph
)

# Smallest fix for the clone error: skip make_learner(fitted=True) on this variant
# and fit the CatBoost model directly on the holdout train split.
X_train_cat = train_part.drop(columns=target_col, errors="ignore")
y_train_cat = train_part[target_col]
X_valid_cat = valid_part.drop(columns=target_col, errors="ignore")
y_valid_cat = valid_part[target_col]

cat_direct = CatBoostClassifier(
    cat_features=cat_features,
    loss_function="MultiClass",
    random_state=random_state,
    verbose=0,
)
cat_direct.fit(X_train_cat, y_train_cat)
valid_pred_cat = cat_direct.predict(X_valid_cat).reshape(-1)
scores["cat_no_tablevectorizer"] = accuracy_score(y_valid_cat, valid_pred_cat)
print(f"Ablation[cat_no_tablevectorizer] {metric_label}: {scores['cat_no_tablevectorizer']}")

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")
