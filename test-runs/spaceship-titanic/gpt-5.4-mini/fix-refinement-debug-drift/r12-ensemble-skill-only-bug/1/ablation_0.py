import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def prep_columns(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", expand=True)
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"])

    if "Name" in out.columns:
        out["NameLength"] = out["Name"].astype("string").str.len()
        out["NameTokens"] = out["Name"].astype("string").str.split().str.len()
        out = out.drop(columns=["Name"])

    bool_cols = [c for c in ["CryoSleep", "VIP"] if c in out.columns]
    for c in bool_cols:
        out[c] = out[c].astype("string").map({"True": 1, "False": 0}).astype(float)

    if "PassengerId" in out.columns:
        pid = out["PassengerId"].astype("string").str.split("_", expand=True)
        out["Group"] = pid[0]
        out["Person"] = pid[1]

    return out


def prep_columns_no_name(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", expand=True)
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"])

    bool_cols = [c for c in ["CryoSleep", "VIP"] if c in out.columns]
    for c in bool_cols:
        out[c] = out[c].astype("string").map({"True": 1, "False": 0}).astype(float)

    if "PassengerId" in out.columns:
        pid = out["PassengerId"].astype("string").str.split("_", expand=True)
        out["Group"] = pid[0]
        out["Person"] = pid[1]

    return out


def prep_columns_minimal(df):
    out = df.copy()
    bool_cols = [c for c in ["CryoSleep", "VIP"] if c in out.columns]
    for c in bool_cols:
        out[c] = out[c].astype("string").map({"True": 1, "False": 0}).astype(float)
    return out


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred_graph = build_graph(data_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred)
    if valid_pred.ndim == 2 and valid_pred.shape[1] > 1:
        valid_pred = valid_pred[:, 1]
    valid_pred = (valid_pred.ravel() >= 0.5)
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy_score: {score}")
    return score


def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    return X_train.skb.apply_func(prep_columns).skb.apply(
        skrub.TableVectorizer()
    ).skb.apply(
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        ),
        y=y_train,
    )


def no_name_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    return X_train.skb.apply_func(prep_columns_no_name).skb.apply(
        skrub.TableVectorizer()
    ).skb.apply(
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        ),
        y=y_train,
    )


def lightgbm_only_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    lgbm_pred = X_train.skb.apply_func(prep_columns).skb.apply(
        skrub.TableVectorizer()
    ).skb.apply(
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        ),
        y=y_train,
    )
    return lgbm_pred


def catboost_only_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    return X_train.skb.apply_func(prep_columns).skb.apply(
        skrub.TableVectorizer()
    ).skb.apply(
        CatBoostClassifier(
            verbose=0,
            random_seed=random_state,
            loss_function="Logloss",
            allow_writing_files=False,
        ),
        y=y_train,
    )


def minimal_prep_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    return X_train.skb.apply_func(prep_columns_minimal).skb.apply(
        skrub.TableVectorizer()
    ).skb.apply(
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        ),
        y=y_train,
    )


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["no_name_feature"] = score_variant("no_name_feature", no_name_graph)
scores["lightgbm_only"] = score_variant("lightgbm_only", lightgbm_only_graph)
scores["catboost_only"] = score_variant("catboost_only", catboost_only_graph)
scores["minimal_prep"] = score_variant("minimal_prep", minimal_prep_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy_score: {best_score}")