import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"
metric_label = "accuracy"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

def add_features(df):
    df = df.copy()
    cabin = df["Cabin"].fillna("X").astype(str).str.split("/")
    df["CabinDeck"] = cabin.str[0]
    df["CabinNum"] = pd.to_numeric(cabin.str[1], errors="coerce")
    df["CabinSide"] = cabin.str[2]
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["GroupSize"] = df["PassengerId"].astype(str).str.split("_").str[0].map(
        df["PassengerId"].astype(str).str.split("_").str[0].value_counts()
    )
    df["NameLen"] = df["Name"].fillna("").astype(str).str.len()
    df["Surname"] = df["Name"].fillna("Unknown").astype(str).str.split().str[-1]
    return df

train_df = add_features(train_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred_graph = build_graph(data_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred).reshape(-1).astype(bool)
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    features = [
        "HomePlanet", "CryoSleep", "CabinDeck", "CabinSide", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "CabinNum", "TotalSpend", "NoSpend", "GroupSize", "NameLen", "Surname"
    ]
    vectorizer = skrub.TableVectorizer()
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )
    return X_train[features].skb.apply(vectorizer).skb.apply(model, y=y_train)

def no_engineered_features_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    features = [
        "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
    ]
    vectorizer = skrub.TableVectorizer()
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )
    return X_train[features].skb.apply(vectorizer).skb.apply(model, y=y_train)

def no_high_cardinality_routing_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    features = [
        "HomePlanet", "CryoSleep", "CabinDeck", "CabinSide", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "CabinNum", "TotalSpend", "NoSpend", "GroupSize", "NameLen"
    ]
    vectorizer = skrub.TableVectorizer(high_cardinality="drop")
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )
    return X_train[features].skb.apply(vectorizer).skb.apply(model, y=y_train)

def simpler_model_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    features = [
        "HomePlanet", "CryoSleep", "CabinDeck", "CabinSide", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "CabinNum", "TotalSpend", "NoSpend", "GroupSize", "NameLen", "Surname"
    ]
    vectorizer = skrub.TableVectorizer()
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )
    return X_train[features].skb.apply(vectorizer).skb.apply(model, y=y_train)

scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["no_engineered_features"] = score_variant("no_engineered_features", no_engineered_features_graph)
scores["drop_high_cardinality"] = score_variant("drop_high_cardinality", no_high_cardinality_routing_graph)
scores["simpler_model"] = score_variant("simpler_model", simpler_model_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")