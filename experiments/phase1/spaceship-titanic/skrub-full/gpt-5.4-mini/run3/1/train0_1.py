
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

def add_features(df):
    df = df.copy()
    cabin = df["Cabin"].astype(str)
    cabin_parts = cabin.str.split("/", expand=True)
    df["CabinDeck"] = cabin_parts[0].replace("nan", np.nan)
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["CabinSide"] = cabin_parts[2].replace("nan", np.nan)
    df["Group"] = df["PassengerId"].astype(str).str.split("_", expand=True)[0]
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[-1, 12, 18, 25, 40, 60, 200], labels=False)
    df["RoomServiceLog"] = np.log1p(df["RoomService"].fillna(0))
    df["FoodCourtLog"] = np.log1p(df["FoodCourt"].fillna(0))
    df["ShoppingMallLog"] = np.log1p(df["ShoppingMall"].fillna(0))
    df["SpaLog"] = np.log1p(df["Spa"].fillna(0))
    df["VRDeckLog"] = np.log1p(df["VRDeck"].fillna(0))
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

lgb_model = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=250,
    learning_rate=0.04,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=20,
    verbose=-1,
)

rf_model = RandomForestClassifier(
    n_estimators=350,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=2,
)

lgb_pred = X_train.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train)
rf_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(rf_model, y=y_train)

lgb_learner = lgb_pred.skb.make_learner(fitted=True)
rf_learner = rf_pred.skb.make_learner(fitted=True)

valid_lgb = np.asarray(lgb_learner.predict({"data": valid_part}))
valid_rf = np.asarray(rf_learner.predict({"data": valid_part}))

if valid_lgb.dtype != bool:
    valid_lgb = valid_lgb >= 0.5
if valid_rf.dtype != bool:
    valid_rf = valid_rf >= 0.5

valid_pred = (valid_lgb.astype(int) + valid_rf.astype(int)) >= 1
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
