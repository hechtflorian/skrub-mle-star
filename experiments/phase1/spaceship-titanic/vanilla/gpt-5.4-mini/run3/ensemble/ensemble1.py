
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostClassifier

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def prep(df):
    df = df.copy()

    # Basic splits
    cabin = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    pid = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["GroupId"] = pid[0]
    df["GroupSize"] = pd.to_numeric(pid[1], errors="coerce")

    # Spending features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)

    # Age features
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 65, 120],
        labels=False,
        include_lowest=True,
    )

    # Cabin number bins
    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 10, 50, 200, 500, 1000, 5000, np.inf],
        labels=False,
        include_lowest=True,
    )

    # Name features
    df["NameLen"] = df["Name"].astype(str).str.len()

    # Missing handling / typing
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "AgeBin", "CabinNumBin"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    for col in ["GroupId"]:
        df[col] = df[col].astype("object").fillna("Missing").astype(str)

    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum", "GroupSize", "TotalSpend", "NameLen"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Drop raw text fields
    df = df.drop(columns=["Cabin", "Name"], errors="ignore")
    return df

train_p = prep(train)
test_p = prep(test)

y = train_p["Transported"].astype(int)
X = train_p.drop(columns=["Transported"])

# Ensure same columns and ordering
X = X.copy()
test_p = test_p[X.columns].copy()

cat_features = [
    c for c in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "AgeBin", "CabinNumBin", "GroupId"]
    if c in X.columns
]

for col in X.columns:
    if col in cat_features:
        X[col] = X[col].astype(str)
        test_p[col] = test_p[col].astype(str)
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        test_p[col] = pd.to_numeric(test_p[col], errors="coerce")

# Smaller, faster ensemble to avoid timeout
seeds = [42, 202]
n_splits = 3

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), len(seeds)))
test_preds = np.zeros((len(test_p), len(seeds)))

base_params = dict(
    loss_function="Logloss",
    iterations=600,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=4.0,
    random_strength=1.0,
    subsample=0.8,
    bootstrap_type="Bernoulli",
    eval_metric="Accuracy",
    verbose=0,
    allow_writing_files=False,
)

cat_indices = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

for m_idx, seed in enumerate(seeds):
    oof = np.zeros(len(X))
    tst = np.zeros(len(test_p))

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(**base_params, random_seed=seed)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_indices,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=50,
        )

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        tst += model.predict_proba(test_p)[:, 1] / n_splits

    oof_preds[:, m_idx] = oof
    test_preds[:, m_idx] = tst

# Simple and fast meta-model
meta = LogisticRegression(max_iter=1000)
meta.fit(oof_preds, y)

# Validation score on OOF predictions
val_proba = meta.predict_proba(oof_preds)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)
final_validation_score = accuracy_score(y, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_final_proba = meta.predict_proba(test_preds)[:, 1]
test_pred = (test_final_proba >= 0.5).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
