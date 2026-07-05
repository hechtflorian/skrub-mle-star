
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

input_dir = "./input"
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))

def feature_engineer(df):
    df = df.copy()
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", n=1, expand=True)
        df["PassengerGroup"] = pd.to_numeric(pid[0], errors="coerce")
        df["PassengerNumber"] = pd.to_numeric(pid[1], errors="coerce")
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", n=2, expand=True)
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype(str).str.split(" ", n=1).str[-1]
        df["NameLen"] = df["Name"].astype(str).str.len()
    df["TotalSpending"] = (
        df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
        .fillna(0)
        .sum(axis=1)
    )
    df["SpendingFlag"] = (df["TotalSpending"] > 0).astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-np.inf, 12, 18, 30, 50, np.inf],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    return df

train_part, valid_part = train_test_split(
    train_df, test_size=0.2, random_state=random_state, stratify=train_df[target_col]
)

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(feature_engineer)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

pred_lgbm = X_train.skb.apply(
    vectorizer,
).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    ),
    y=y_train,
)

pred_cat = X_train.skb.apply(
    vectorizer,
).skb.apply(
    CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=random_state,
        verbose=0,
    ),
    y=y_train,
)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

valid_pred_lgbm = np.asarray(learner_lgbm.predict({"data": valid_part})).ravel()
valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part})).ravel()

if valid_pred_lgbm.dtype.kind not in "fc":
    valid_pred_lgbm = valid_pred_lgbm.astype(float)
if valid_pred_cat.dtype.kind not in "fc":
    valid_pred_cat = valid_pred_cat.astype(float)

blended_pred = 0.5 * valid_pred_lgbm + 0.5 * valid_pred_cat
final_pred = blended_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], final_pred)
print(f"Final Validation Performance: {final_validation_score}")
