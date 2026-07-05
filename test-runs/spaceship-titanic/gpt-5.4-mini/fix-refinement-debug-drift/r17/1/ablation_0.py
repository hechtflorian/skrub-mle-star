
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

random_state = 42
test_size = 0.2
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    final_validation_score = accuracy_score(
        valid_part[target_col].astype(int), valid_pred.astype(int)
    )
    print(f"Ablation[{variant_name}] accuracy: {final_validation_score}")
    return final_validation_score


def preprocess_func(df):
    out = df.copy()

    for col in ["CryoSleep", "VIP", "Transported"]:
        if col in out.columns:
            out[col] = out[col].map({True: 1, False: 0, "True": 1, "False": 0})

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype(str).str.split("/", expand=True)
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"])

    if "Name" in out.columns:
        out["NameLen"] = out["Name"].astype(str).str.len()
        out = out.drop(columns=["Name"])

    return out


def baseline_graph(data_train):
    data_train = data_train.skb.apply_func(preprocess_func)
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_state=random_state,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)


def lgbm_graph(data_train):
    data_train = data_train.skb.apply_func(preprocess_func)
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)


def no_feature_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_state=random_state,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["lgbm"] = score_variant("lgbm", lgbm_graph)
scores["no_feature"] = score_variant("no_feature", no_feature_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {best_score}")
print(f"Final Validation Performance: {best_score}")
