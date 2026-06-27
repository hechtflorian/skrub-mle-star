
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cabin = out["Cabin"].fillna("U/U/U").str.split("/", expand=True)
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

    out["CabinGroupSize"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["NameLength"] = out["Name"].fillna("").str.len()
    out["Surname"] = out["Name"].fillna("").str.split().str[-1].astype(str)

    out["LuxurySpending"] = out[["Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["BasicSpending"] = out[["RoomService", "FoodCourt", "ShoppingMall"]].fillna(0).sum(axis=1)

    out["NoSpending"] = (out["Spending"] == 0).astype(int)

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)

X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].astype(int).skb.mark_as_y()

cat_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "AgeGroup",
    "CabinGroupSize",
    "Surname",
]

for c in cat_cols:
    if c in train_part.columns:
        pass

vectorizer = skrub.TableVectorizer()
model = CatBoostClassifier(
    iterations=1200,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=0,
    random_seed=42,
    allow_writing_files=False,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)

valid_pred = learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col].astype(int), valid_pred.astype(int))
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)
X_full = data_full_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].astype(int).skb.mark_as_y()

full_pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred_graph.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df}).astype(bool)
submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
