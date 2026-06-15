
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure catboost is available
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
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



def prep(df):
    df = df.copy()

    # --- Cabin parsing (strengthened) ---
    cabin_raw = df["Cabin"].astype("string")
    df["CabinMissing"] = cabin_raw.isna().astype(int)

    cabin_split = cabin_raw.fillna("Missing/Missing/Missing").str.split("/", expand=True)
    df["Deck"] = cabin_split[0].fillna("Missing")
    df["Num"] = cabin_split[1]
    df["Side"] = cabin_split[2].fillna("Missing")

    df["CabinNum"] = pd.to_numeric(df["Num"], errors="coerce")
    df["CabinNumMissing"] = df["CabinNum"].isna().astype(int)

    # Structured cabin signals
    df["CabinNumParity"] = np.where(df["CabinNum"].notna(), (df["CabinNum"] % 2).astype("Int64"), pd.NA)
    df["CabinNumDecile"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000, np.inf],
        labels=False,
        include_lowest=True,
    )
    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 50, 200, 500, 1000, 2000, 5000, np.inf],
        labels=False,
        include_lowest=True,
    )
    df["CabinNumLog"] = np.log1p(df["CabinNum"])

    # Cabin-derived cross features
    df["DeckSide"] = df["Deck"].astype("string").fillna("Missing") + "_" + df["Side"].astype("string").fillna("Missing")
    df["DeckNumBin"] = df["Deck"].astype("string").fillna("Missing") + "_" + df["CabinNumBin"].astype("string").fillna("Missing")

    # --- Passenger group features ---
    pid_split = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["Group"] = pid_split[0]
    df["GroupNum"] = pid_split[1]

    # Group size and relative position within group
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("size")
    df["GroupNum"] = pd.to_numeric(df["GroupNum"], errors="coerce")
    df["GroupPos"] = df["GroupNum"]

    # Group-level cabin consistency / shared structure
    df["GroupDeckConsistent"] = df.groupby("Group")["Deck"].transform(lambda s: int(s.nunique(dropna=False) == 1))
    df["GroupSideConsistent"] = df.groupby("Group")["Side"].transform(lambda s: int(s.nunique(dropna=False) == 1))
    df["GroupCabinMissingRate"] = df.groupby("Group")["CabinMissing"].transform("mean")
    df["GroupCabinNumMean"] = df.groupby("Group")["CabinNum"].transform("mean")
    df["GroupCabinNumStd"] = df.groupby("Group")["CabinNum"].transform("std")

    # --- Keep existing non-cabin features, but simplify noisy spending block ---
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spending_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Preserve a compact signal rather than multiple redundant spending-derived features
    df["SpendingTotal"] = df[spending_cols].sum(axis=1)
    df["SpendingAny"] = (df[spending_cols].fillna(0).sum(axis=1) > 0).astype(int)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 70, 120],
        labels=False,
        include_lowest=True,
    )

    df["NameLen"] = df["Name"].astype(str).str.len()

    # Fill missing values
    for col in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "DeckSide", "DeckNumBin"]:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum", "NameLen", "SpendingTotal", "CabinNumLog", "GroupCabinNumMean", "GroupCabinNumStd"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ["AgeBin", "CabinNumBin", "CabinNumParity", "CabinNumDecile"]:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    for col in ["GroupSize", "GroupPos", "GroupDeckConsistent", "GroupSideConsistent", "GroupCabinMissingRate", "CabinMissing", "CabinNumMissing", "SpendingAny"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Keep CatBoost-friendly categorical fields as strings
    for col in ["Group", "GroupNum", "Side", "DeckSide"]:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    df = df.drop(columns=["Cabin", "Name", "Num"], errors="ignore")
    return df



train_p = prep(train)
test_p = prep(test)

X = train_p.drop(columns=["Transported"])
y = train_p["Transported"].astype(int)

cat_features = [
    "HomePlanet", "CryoSleep", "Destination", "VIP",
    "Deck", "Side", "Group", "GroupNum", "AgeBin", "CabinNumBin"
]

for col in X.columns:
    if col in cat_features:
        X[col] = X[col].astype(str)
        test_p[col] = test_p[col].astype(str)
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if col in test_p.columns:
            test_p[col] = pd.to_numeric(test_p[col], errors="coerce")

# Align columns just in case
test_p = test_p[X.columns]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    loss_function="Logloss",
    depth=8,
    learning_rate=0.05,
    iterations=2000,
    random_seed=42,
    verbose=0
)

cat_indices = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

model.fit(X_train, y_train, cat_features=cat_indices)

val_pred = model.predict(X_val).astype(int).ravel()
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

model.fit(X, y, cat_features=cat_indices)
test_pred = model.predict(test_p).astype(bool).ravel()

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
