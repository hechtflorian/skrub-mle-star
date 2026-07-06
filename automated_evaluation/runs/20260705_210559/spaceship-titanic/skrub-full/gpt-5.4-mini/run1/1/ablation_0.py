
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
test_size = 0.2
target_col = "Transported"
metric_fn = accuracy_score
metric_label = "accuracy"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

def fe_func(df):
    out = df.copy()
    cabin_parts = out["Cabin"].astype(str).str.split("/", expand=True)
    out["CabinDeck"] = cabin_parts[0]
    out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    out["CabinSide"] = cabin_parts[2]

    name_parts = out["Name"].astype(str).str.split(" ", n=1, expand=True)
    out["FirstName"] = name_parts[0]
    out["LastName"] = name_parts[1]

    out["TotalSpend"] = (
        out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
        .fillna(0)
        .sum(axis=1)
    )
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    return out

def build_baseline(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def build_no_fe(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def build_with_fe(data_train):
    data_train = data_train.skb.apply_func(fe_func)
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=0,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = metric_fn(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

scores = {}
scores["baseline"] = score_variant("baseline", build_baseline)
scores["no_fe"] = score_variant("no_fe", build_no_fe)
scores["with_fe"] = score_variant("with_fe", build_with_fe)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
final_validation_score = best_score
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
print(f"Final Validation Performance: {final_validation_score}")
