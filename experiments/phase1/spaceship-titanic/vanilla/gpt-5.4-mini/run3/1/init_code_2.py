
import os
import warnings
import subprocess
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure required package is available
try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "AgeBin", "CabinNumBin"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum", "GroupNum", "NameLen", "Spending", "SpendingLog"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.drop(columns=["Cabin", "Name"], errors="ignore")
    return df

train_p = prep(train)
test_p = prep(test)

X = train_p.drop(columns=["Transported"])
y = train_p["Transported"].astype(int)

X = pd.get_dummies(X, dummy_na=False)
test_p = pd.get_dummies(test_p, dummy_na=False)
test_p = test_p.reindex(columns=X.columns, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

model.fit(X, y)
test_pred = model.predict(test_p).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)
