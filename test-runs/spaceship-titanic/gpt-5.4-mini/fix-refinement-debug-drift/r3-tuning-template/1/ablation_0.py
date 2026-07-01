
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"
metric_label = "accuracy"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def fe_func(df):
    out = df.copy()
    cabin = out["Cabin"].fillna("Unknown/Unknown/Unknown").astype(str).str.split("/", expand=True)
    out["CabinDeck"] = cabin[0]
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["CabinSide"] = cabin[2]
    out["TotalSpend"] = (
        out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
        .fillna(0)
        .sum(axis=1)
    )
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["GroupSize"] = out["PassengerId"].astype(str).str.split("_").str[0].map(
        out["PassengerId"].astype(str).str.split("_").str[0].value_counts()
    )
    return out


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
    encoder = skrub.TableVectorizer()
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return X.skb.apply_func(fe_func).skb.apply(encoder).skb.apply(model, y=y)


def no_fe_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return X.skb.apply(encoder).skb.apply(model, y=y)


def alt_encoder_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer(high_cardinality="drop")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return X.skb.apply_func(fe_func).skb.apply(encoder).skb.apply(model, y=y)


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["no_fe"] = score_variant("no_fe", no_fe_graph)
scores["alt_encoder"] = score_variant("alt_encoder", alt_encoder_graph)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
final_validation_score = best_score
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
print(f"Final Validation Performance: {final_validation_score}")
