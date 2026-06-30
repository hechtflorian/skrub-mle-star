
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"

def add_features(df):
    out = df.copy()

    cabin = out["Cabin"].fillna("Missing/Missing/Missing").astype(str).str.split("/", expand=True)
    out["CabinDeck"] = cabin[0].fillna("Missing")
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["CabinSide"] = cabin[2].fillna("Missing")

    for col in ["CryoSleep", "VIP"]:
        if col in out.columns:
            out[col] = out[col].astype("object").where(out[col].notna(), "Missing")

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["TotalSpending"] = out[spend_cols].fillna(0).sum(axis=1)
    out["AvgSpending"] = out["TotalSpending"] / len(spend_cols)
    out["NoSpending"] = (out["TotalSpending"] == 0).astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-np.inf, 12, 18, 30, 45, 60, np.inf],
        labels=["child", "teen", "young_adult", "adult", "mid_age", "senior"],
    ).astype("object")
    out["AgeGroup"] = out["AgeGroup"].astype("object").where(out["AgeGroup"].notna(), "Missing")

    out["RoomService_log"] = np.log1p(out["RoomService"].fillna(0))
    out["FoodCourt_log"] = np.log1p(out["FoodCourt"].fillna(0))
    out["ShoppingMall_log"] = np.log1p(out["ShoppingMall"].fillna(0))
    out["Spa_log"] = np.log1p(out["Spa"].fillna(0))
    out["VRDeck_log"] = np.log1p(out["VRDeck"].fillna(0))

    group = out["PassengerId"].astype(str).str.split("_", expand=True)
    out["GroupId"] = group[0].fillna("Missing")
    out["GroupNum"] = pd.to_numeric(group[1], errors="coerce")

    out["NameLength"] = out["Name"].fillna("").astype(str).str.len()

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col].astype(int)
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)

X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=0,
    random_seed=42,
    allow_writing_files=False,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)

learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})

final_validation_score = accuracy_score(
    valid_part[target_col].astype(int),
    np.asarray(valid_pred).astype(int),
)

print(f"Final Validation Performance: {final_validation_score}")
