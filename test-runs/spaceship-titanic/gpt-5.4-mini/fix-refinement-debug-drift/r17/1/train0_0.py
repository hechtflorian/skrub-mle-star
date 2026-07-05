
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

for df in [train_df]:
    df["CabinDeck"] = df["Cabin"].fillna("X").astype(str).str.split("/").str[0]
    df["CabinNum"] = pd.to_numeric(df["Cabin"].fillna("X").astype(str).str.split("/").str[1], errors="coerce")
    df["CabinSide"] = df["Cabin"].fillna("X").astype(str).str.split("/").str[2]
    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
    df["GroupSize"] = df["Group"].map(df.groupby("Group").size())
    df["Surname"] = df["Name"].fillna("Unknown").astype(str).str.split(" ").str[-1]
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[-1, 12, 18, 25, 35, 50, 65, 200], labels=False)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = accuracy_score(
    valid_part[target_col].astype(int),
    np.asarray(valid_pred).astype(int),
)
print(f"Final Validation Performance: {final_validation_score}")
