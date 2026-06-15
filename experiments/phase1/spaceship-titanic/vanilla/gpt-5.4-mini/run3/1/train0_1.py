
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure required packages are available
from importlib.util import find_spec

if find_spec("catboost") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
if find_spec("lightgbm") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def prep(df):
    df = df.copy()

    cabin = df["Cabin"].astype(str)
    cabin_split = cabin.str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    pid_split = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["Group"] = pid_split[0]
    df["GroupNum"] = pd.to_numeric(pid_split[1], errors="coerce")

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spending_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Spending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["Spending"].fillna(0) == 0).astype(int)
    df["SpendingLog"] = np.log1p(df["Spending"].fillna(0))

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 30, 50, 120],
        labels=False,
        include_lowest=True,
    )

    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 50, 200, 500, 1000, 2000, 5000, np.inf],
        labels=False,
        include_lowest=True,
    )

    df["NameLen"] = df["Name"].astype(str).str.len()
    df["IsAlone"] = (df["GroupNum"] == 1).astype(int)

    cat_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "Group",
        "AgeBin",
        "CabinNumBin",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    num_cols = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "CabinNum",
        "GroupNum",
        "NameLen",
        "Spending",
        "SpendingLog",
        "NoSpending",
        "IsAlone",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.drop(columns=["Cabin", "Name"], errors="ignore")
    return df


train_p = prep(train)
test_p = prep(test)

X = train_p.drop(columns=["Transported"])
y = train_p["Transported"].astype(int)

X_cat = X.copy()
test_cat = test_p.copy()

cat_features = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "Group",
    "AgeBin",
    "CabinNumBin",
]
for col in X_cat.columns:
    if col in cat_features:
        X_cat[col] = X_cat[col].astype(str)
        test_cat[col] = test_cat[col].astype(str)
    else:
        X_cat[col] = pd.to_numeric(X_cat[col], errors="coerce")
        test_cat[col] = pd.to_numeric(test_cat[col], errors="coerce")

test_cat = test_cat[X_cat.columns]

X_lgb = pd.get_dummies(X_cat, dummy_na=False)
test_lgb = pd.get_dummies(test_cat, dummy_na=False)
test_lgb = test_lgb.reindex(columns=X_lgb.columns, fill_value=0)

X_train_lgb, X_val_lgb, y_train, y_val = train_test_split(
    X_lgb, y, test_size=0.2, random_state=42, stratify=y
)

X_train_cat, X_val_cat, _, _ = train_test_split(
    X_cat, y, test_size=0.2, random_state=42, stratify=y
)

lgb_model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

cat_model = CatBoostClassifier(
    loss_function="Logloss",
    depth=8,
    learning_rate=0.05,
    iterations=2000,
    random_seed=42,
    verbose=0,
)

lgb_model.fit(X_train_lgb, y_train)
cat_indices = [X_cat.columns.get_loc(c) for c in cat_features if c in X_cat.columns]
cat_model.fit(X_train_cat, y_train, cat_features=cat_indices)

val_pred_lgb = lgb_model.predict(X_val_lgb).astype(int).ravel()
val_pred_cat = cat_model.predict(X_val_cat).astype(int).ravel()

val_pred_ens = ((val_pred_lgb + val_pred_cat) >= 1).astype(int)
val_acc = accuracy_score(y_val, val_pred_ens)
print(f"Final Validation Performance: {val_acc}")

lgb_model.fit(X_lgb, y)
cat_model.fit(X_cat, y, cat_features=cat_indices)

test_pred_lgb = lgb_model.predict(test_lgb).astype(int).ravel()
test_pred_cat = cat_model.predict(test_cat).astype(int).ravel()
test_pred = ((test_pred_lgb + test_pred_cat) >= 1).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
