
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

def add_features(df):
    df = df.copy()
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-np.inf, 12, 18, 30, 50, np.inf],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    ).astype("object")
    for col in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin", "Name", "AgeGroup"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["Spent"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
    df["NoSpending"] = (df["Spent"] == 0).astype(int)
    df = df.drop(columns=["Cabin"], errors="ignore")
    return df

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

train_part = add_features(train_part)
valid_part = add_features(valid_part)

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = RandomForestClassifier(
    n_estimators=300,
    random_state=random_state,
    n_jobs=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

if hasattr(valid_pred, "dtype") and str(getattr(valid_pred, "dtype")) == "bool":
    valid_pred_label = valid_pred
else:
    valid_pred_arr = np.asarray(valid_pred)
    if valid_pred_arr.dtype == bool:
        valid_pred_label = valid_pred_arr
    elif valid_pred_arr.ndim > 1 and valid_pred_arr.shape[1] > 1:
        valid_pred_label = valid_pred_arr[:, 1] > 0.5
    else:
        valid_pred_label = valid_pred_arr.astype(float) > 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred_label)
print(f"Final Validation Performance: {final_validation_score}")
