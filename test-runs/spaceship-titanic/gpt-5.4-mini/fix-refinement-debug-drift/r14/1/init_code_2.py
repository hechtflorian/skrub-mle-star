
import os
import re
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def fe_func(df):
    df = df.copy()

    cabin = df["Cabin"].astype("string")
    cabin_split = cabin.str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["TotalSpent"] = df[spend_cols].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpent"] == 0).astype(int)
    df["IsChild"] = (pd.to_numeric(df["Age"], errors="coerce") < 13).astype(int)

    name = df["Name"].astype("string")
    surname = name.str.split().str[-1]
    group_id = df["PassengerId"].astype("string").str.split("_").str[0]
    df["GroupSize"] = group_id.map(group_id.value_counts())

    return df

train_df = fe_func(train_df)
test_df = fe_func(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

feat_cols = [
    "HomePlanet", "CryoSleep", "Destination", "Age", "VIP",
    "Deck", "Num", "Side", "TotalSpent", "NoSpend", "IsChild", "GroupSize"
]

vectorizer = skrub.TableVectorizer()
model = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    ),
)

pred_chain = X_train[feat_cols].skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred_chain.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)
if valid_pred.ndim > 1:
    valid_pred = valid_pred[:, -1]
valid_labels = (valid_pred >= 0.5).astype(bool)
final_validation_score = accuracy_score(valid_part[target_col], valid_labels)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_chain = X_full[feat_cols].skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred_chain.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred)
if test_pred.ndim > 1:
    test_pred = test_pred[:, -1]
test_labels = (test_pred >= 0.5).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_labels,
    }
)
submission.to_csv("submission.csv", index=False)
