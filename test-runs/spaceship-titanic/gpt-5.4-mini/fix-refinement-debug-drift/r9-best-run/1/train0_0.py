
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def add_features(df):
    df = df.copy()
    cabin = df["Cabin"].fillna("X").astype(str).str.split("/")
    df["CabinDeck"] = cabin.str[0]
    df["CabinNum"] = pd.to_numeric(cabin.str[1], errors="coerce")
    df["CabinSide"] = cabin.str[2]
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["GroupSize"] = df["PassengerId"].astype(str).str.split("_").str[0].map(df["PassengerId"].astype(str).str.split("_").str[0].value_counts())
    df["NameLen"] = df["Name"].fillna("").astype(str).str.len()
    df["Surname"] = df["Name"].fillna("Unknown").astype(str).str.split().str[-1]
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
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

pred_graph = X_train[features].skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).reshape(-1)
valid_pred = pd.Series(valid_pred).astype(bool).to_numpy()

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
