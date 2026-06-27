import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "Transported"
metric_name = "accuracy"


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col].astype(int),
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cabin = out["Cabin"].fillna("U/U/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["Spending"] = out[spending_cols].fillna(0).sum(axis=1)
    out["HasSpending"] = (out["Spending"] > 0).astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 30, 50, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("object")

    out["PassengerGroup"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["NameLength"] = out["Name"].fillna("").astype(str).str.len()
    out["Surname"] = out["Name"].fillna("").astype(str).str.split().str[-1]

    out["LuxurySpending"] = out[["Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["BasicSpending"] = out[["RoomService", "FoodCourt", "ShoppingMall"]].fillna(0).sum(axis=1)
    out["NoSpending"] = (out["Spending"] == 0).astype(int)

    out["FamilyNameKey"] = (
        out["Surname"].fillna("Unknown").astype(str)
        + "_"
        + out["PassengerGroup"].fillna("Unknown").astype(str)
    )
    return out


def add_features_reduced(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cabin = out["Cabin"].fillna("U/U/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Side"] = cabin[2]
    out["Spending"] = out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["HasSpending"] = (out["Spending"] > 0).astype(int)
    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 30, 50, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("object")
    out["PassengerGroup"] = out["PassengerId"].astype(str).str.split("_").str[0]
    return out


def add_features_no_cabin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["Spending"] = out[spending_cols].fillna(0).sum(axis=1)
    out["HasSpending"] = (out["Spending"] > 0).astype(int)
    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 30, 50, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("object")
    out["PassengerGroup"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["NameLength"] = out["Name"].fillna("").astype(str).str.len()
    return out


def build_graph(data_train, fe_func):
    data_fe = data_train.skb.apply_func(fe_func)
    X = data_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data_fe[target_col].astype(int).skb.mark_as_y()

    vectorizer_lgbm = skrub.TableVectorizer()
    vectorizer_cat = skrub.TableVectorizer()

    lgbm_model = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    cat_model = CatBoostClassifier(
        iterations=1200,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Accuracy",
        verbose=0,
        random_seed=42,
        allow_writing_files=False,
    )

    lgbm_graph = X.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y)
    cat_graph = X.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y)

    lgbm_learner = lgbm_graph.skb.make_learner(fitted=True)
    cat_learner = cat_graph.skb.make_learner(fitted=True)

    pred_lgbm = lgbm_learner.predict({"data": valid_part}).astype(float)
    pred_cat = cat_learner.predict({"data": valid_part}).astype(float)
    pred_ensemble = ((pred_lgbm + pred_cat) / 2.0 >= 0.5).astype(int)
    return pred_ensemble


def score_variant(variant_name, fe_func):
    data_train = skrub.var("data", train_part)
    valid_pred = build_graph(data_train, fe_func)
    score = accuracy_score(valid_part[target_col].astype(int), valid_pred)
    print(f"Ablation[{variant_name}] {metric_name}: {score}")
    return score


scores = {}
scores["baseline"] = score_variant("baseline", add_features)
scores["reduced_fe"] = score_variant("reduced_fe", add_features_reduced)
scores["no_cabin_features"] = score_variant("no_cabin_features", add_features_no_cabin)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_name}: {best_score}")