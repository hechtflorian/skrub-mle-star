import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = os.path.join("./input", "train.csv")
train = pd.read_csv(train_path)

def prep(df):
    df = df.copy()

    cabin = df["Cabin"].astype(str)
    cabin_split = cabin.str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = cabin_split[1]
    df["Side"] = cabin_split[2]

    pid_split = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["Group"] = pid_split[0]
    df["GroupNum"] = pid_split[1]

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spending_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Spending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["Spending"].fillna(0) == 0).astype(int)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 70, 120],
        labels=False,
        include_lowest=True,
    )

    df["CabinNum"] = pd.to_numeric(df["Num"], errors="coerce")
    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 50, 200, 500, 1000, 2000, 5000, np.inf],
        labels=False,
        include_lowest=True,
    )

    df["NameLen"] = df["Name"].astype(str).str.len()

    for col in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum"]:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum", "NameLen", "Spending"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ["AgeBin", "CabinNumBin"]:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    df = df.drop(columns=["Cabin", "Name", "Num"], errors="ignore")
    return df

train_p = prep(train)

X = train_p.drop(columns=["Transported"])
y = train_p["Transported"].astype(int)

cat_features_base = [
    "HomePlanet", "CryoSleep", "Destination", "VIP",
    "Deck", "Side", "Group", "GroupNum", "AgeBin", "CabinNumBin"
]

for col in X.columns:
    if col in cat_features_base:
        X[col] = X[col].astype(str)
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def fit_eval(Xtr, ytr, Xva, yva, cat_features, desc):
    cat_indices = [Xtr.columns.get_loc(c) for c in cat_features if c in Xtr.columns]
    model = CatBoostClassifier(
        loss_function="Logloss",
        depth=8,
        learning_rate=0.05,
        iterations=2000,
        random_seed=42,
        verbose=0
    )
    model.fit(Xtr, ytr, cat_features=cat_indices)
    pred = model.predict(Xva).astype(int).ravel()
    acc = accuracy_score(yva, pred)
    print(f"{desc}: Validation Accuracy = {acc:.5f}")
    return acc

# Baseline
baseline_acc = fit_eval(X_train, y_train, X_val, y_val, cat_features_base, "Baseline")

# Ablation 1: Disable engineered spending features
X_no_spend = X.drop(columns=["Spending", "NoSpending"], errors="ignore")
Xtr1, Xva1, ytr1, yva1 = train_test_split(
    X_no_spend, y, test_size=0.2, random_state=42, stratify=y
)
acc_no_spend = fit_eval(Xtr1, ytr1, Xva1, yva1, cat_features_base, "Ablation - remove Spending/NoSpending")

# Ablation 2: Disable cabin-derived features
X_no_cabin = X.drop(columns=["Deck", "Side", "CabinNum", "CabinNumBin"], errors="ignore")
cat_features_no_cabin = [c for c in cat_features_base if c not in ["Deck", "Side", "CabinNumBin"]]
Xtr2, Xva2, ytr2, yva2 = train_test_split(
    X_no_cabin, y, test_size=0.2, random_state=42, stratify=y
)
acc_no_cabin = fit_eval(Xtr2, ytr2, Xva2, yva2, cat_features_no_cabin, "Ablation - remove Cabin-derived features")

# Report impact
drops = {
    "Remove Spending/NoSpending": baseline_acc - acc_no_spend,
    "Remove Cabin-derived features": baseline_acc - acc_no_cabin,
}
most_important = max(drops, key=drops.get)

print("\nPerformance drop relative to baseline:")
for k, v in drops.items():
    print(f"{k}: {v:+.5f}")

print(f"\nMost important part for performance: {most_important} (largest accuracy drop)")