
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



def prep(df, train_ref=None):
    df = df.copy()
    ref = train_ref.copy() if train_ref is not None else df

    # --- Base parsing ---
    cabin = df["Cabin"].astype(str)
    cabin_split = cabin.str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = cabin_split[1]
    df["Side"] = cabin_split[2]

    pid_split = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["Group"] = pid_split[0]
    df["GroupNum"] = pid_split[1]

    # --- Numeric spending / age / cabin number ---
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spending_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c in ref.columns:
            ref[c] = pd.to_numeric(ref[c], errors="coerce")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if "Age" in ref.columns:
        ref["Age"] = pd.to_numeric(ref["Age"], errors="coerce")

    df["CabinNum"] = pd.to_numeric(df["Num"], errors="coerce")

    # --- Missingness features ---
    miss_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum", "Age"] + spending_cols
    df["MissingCount"] = df[miss_cols].isna().sum(axis=1)

    # --- Spending features (keep simple) ---
    df["Spending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["Spending"].fillna(0) == 0).astype(int)
    df["LogSpending"] = np.log1p(df["Spending"].fillna(0))

    # --- Family / passenger group features ---
    group_size = df.groupby("Group")["PassengerId"].transform("size")
    df["GroupSize"] = group_size.astype(float)

    group_spend_mean = df.groupby("Group")["Spending"].transform("mean")
    group_spend_std = df.groupby("Group")["Spending"].transform("std")
    df["GroupSpendingMean"] = group_spend_mean
    df["GroupSpendingStd"] = group_spend_std.fillna(0)

    # within-group consistency proxies
    df["SameGroupNoSpending"] = df.groupby("Group")["NoSpending"].transform("sum")
    df["SameGroupCryoSleep"] = df.groupby("Group")["CryoSleep"].transform(lambda s: s.astype(str).eq("True").sum() if len(s) else 0)

    # --- Binned features ---
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 70, 120],
        labels=False,
        include_lowest=True,
    )
    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 50, 200, 500, 1000, 2000, 5000, np.inf],
        labels=False,
        include_lowest=True,
    )

    # --- Name features ---
    df["NameLen"] = df["Name"].astype(str).str.len()

    # --- Dataset-level leakage-safe frequency / co-occurrence features ---
    def _add_freq_features(frame, reference):
        frame = frame.copy()

        freq_cols = ["Deck", "Side", "HomePlanet", "Destination", "Group", "GroupNum"]
        for c in freq_cols:
            if c in frame.columns:
                vc = reference[c].astype(str).fillna("Missing").value_counts(dropna=False)
                frame[f"{c}_Freq"] = frame[c].astype(str).fillna("Missing").map(vc).fillna(0).astype(float)
                frame[f"{c}_FreqLog"] = np.log1p(frame[f"{c}_Freq"])

        pair_cols = [("Deck", "Side"), ("HomePlanet", "Destination"), ("Deck", "HomePlanet"), ("Side", "Destination")]
        for c1, c2 in pair_cols:
            if c1 in frame.columns and c2 in frame.columns:
                pair_key_ref = (
                    reference[c1].astype(str).fillna("Missing") + "_" + reference[c2].astype(str).fillna("Missing")
                )
                pair_vc = pair_key_ref.value_counts(dropna=False)

                pair_key = frame[c1].astype(str).fillna("Missing") + "_" + frame[c2].astype(str).fillna("Missing")
                col_name = f"{c1}_{c2}_Freq"
                frame[col_name] = pair_key.map(pair_vc).fillna(0).astype(float)
                frame[f"{col_name}Log"] = np.log1p(frame[col_name])

        # simple co-occurrence counts within passenger groups
        if "Group" in frame.columns:
            for c in ["Deck", "Side"]:
                if c in frame.columns:
                    grp_nunique = frame.groupby("Group")[c].transform(lambda s: s.astype(str).nunique(dropna=False))
                    frame[f"Group_{c}_NUnique"] = grp_nunique.astype(float)

        return frame

    df = _add_freq_features(df, ref)

    # --- Fill missing values ---
    cat_fill_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum", "AgeBin", "CabinNumBin"]
    for col in cat_fill_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing").astype(str)

    num_fill_cols = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "CabinNum", "NameLen", "Spending", "LogSpending",
        "MissingCount", "GroupSize", "GroupSpendingMean", "GroupSpendingStd",
        "SameGroupNoSpending", "SameGroupCryoSleep",
        "Deck_Freq", "Deck_FreqLog", "Side_Freq", "Side_FreqLog",
        "HomePlanet_Freq", "HomePlanet_FreqLog", "Destination_Freq", "Destination_FreqLog",
        "Group_Freq", "Group_FreqLog", "GroupNum_Freq", "GroupNum_FreqLog",
        "Deck_Side_Freq", "Deck_Side_FreqLog", "HomePlanet_Destination_Freq", "HomePlanet_Destination_FreqLog",
        "Deck_HomePlanet_Freq", "Deck_HomePlanet_FreqLog", "Side_Destination_Freq", "Side_Destination_FreqLog",
        "Group_Deck_NUnique", "Group_Side_NUnique",
    ]
    for col in num_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # keep original categorical columns as strings for CatBoost
    for col in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum"]:
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
