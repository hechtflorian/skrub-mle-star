import os
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


def add_features_full(df):
    out = df.copy()

    cabin = out["Cabin"].fillna("Z/0/Z").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = cabin[1]
    out["Side"] = cabin[2]

    spend_cols = ["Spa", "VRDeck", "RoomService", "FoodCourt", "ShoppingMall"]
    for col in spend_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["TotalSpend"] = out[spend_cols].sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)

    age = pd.to_numeric(out["Age"], errors="coerce")
    out["Age"] = age
    out["IsMinor"] = (age.fillna(age.median()) < 18).astype(int)

    out["CabinNum"] = pd.to_numeric(out["Num"], errors="coerce")
    out["CabinNum"] = out["CabinNum"].replace([np.inf, -np.inf], np.nan)

    out["HasName"] = out["Name"].notna().astype(int)

    group_key = out["PassengerId"].astype(str).str.split("_").str[0]
    out["GroupSize"] = group_key.map(group_key.value_counts())
    out["GroupSize"] = pd.to_numeric(out["GroupSize"], errors="coerce").fillna(1)

    return out


def add_features_no_group(df):
    out = df.copy()

    cabin = out["Cabin"].fillna("Z/0/Z").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = cabin[1]
    out["Side"] = cabin[2]

    spend_cols = ["Spa", "VRDeck", "RoomService", "FoodCourt", "ShoppingMall"]
    for col in spend_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["TotalSpend"] = out[spend_cols].sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)

    age = pd.to_numeric(out["Age"], errors="coerce")
    out["Age"] = age
    out["IsMinor"] = (age.fillna(age.median()) < 18).astype(int)

    out["CabinNum"] = pd.to_numeric(out["Num"], errors="coerce")
    out["CabinNum"] = out["CabinNum"].replace([np.inf, -np.inf], np.nan)

    out["HasName"] = out["Name"].notna().astype(int)
    return out


def add_features_minimal(df):
    out = df.copy()

    spend_cols = ["Spa", "VRDeck", "RoomService", "FoodCourt", "ShoppingMall"]
    for col in spend_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["TotalSpend"] = out[spend_cols].sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)

    age = pd.to_numeric(out["Age"], errors="coerce")
    out["Age"] = age
    out["IsMinor"] = (age.fillna(age.median()) < 18).astype(int)
    return out


def score_variant(variant_name, fe_func, use_cat=True, include_lgb=True):
    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(fe_func)

    X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    preds = []

    if include_lgb:
        vectorizer_lgb = skrub.TableVectorizer()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        pred_lgb = X_train.skb.apply(vectorizer_lgb).skb.apply(lgb_model, y=y_train)
        learner_lgb = pred_lgb.skb.make_learner(fitted=True)
        valid_pred_lgb = np.asarray(learner_lgb.predict({"data": valid_part})).ravel()
        preds.append(valid_pred_lgb)

    if use_cat:
        vectorizer_cat = skrub.TableVectorizer()
        cat_model = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=random_state,
            verbose=0,
            allow_writing_files=False,
        )
        pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)
        learner_cat = pred_cat.skb.make_learner(fitted=True)
        valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part})).ravel()
        preds.append(valid_pred_cat)

    if len(preds) == 1:
        valid_pred = preds[0]
    else:
        valid_pred = 0.55 * preds[0] + 0.45 * preds[1]

    valid_labels = (valid_pred > 0.5).astype(int)
    score = accuracy_score(valid_part[target_col].astype(int), valid_labels)
    print(f"Ablation[{variant_name}] accuracy_score: {score}")
    return score


scores = {}
scores["baseline_full"] = score_variant("baseline_full", add_features_full, use_cat=True, include_lgb=True)
scores["no_group_size"] = score_variant("no_group_size", add_features_no_group, use_cat=True, include_lgb=True)
scores["minimal_features"] = score_variant("minimal_features", add_features_minimal, use_cat=True, include_lgb=True)
scores["lgb_only_full"] = score_variant("lgb_only_full", add_features_full, use_cat=False, include_lgb=True)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy_score: {best_score}")