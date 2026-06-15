
import os
import sys
import warnings
import subprocess

warnings.filterwarnings("ignore")

# Ensure catboost is available
try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# -----------------------------
# Feature engineering
# -----------------------------
def make_features(df, freq_maps=None):
    df = df.copy()

    # Basic parsing
    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    pid_split = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["Group"] = pid_split[0]
    df["GroupNum"] = pid_split[1]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Spending"] = df[spend_cols].sum(axis=1)
    df["LogSpending"] = np.log1p(df["Spending"].fillna(0))
    df["AnyLuxury"] = (df[spend_cols].fillna(0).sum(axis=1) > 0).astype(int)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 70, 120],
        labels=False,
        include_lowest=True,
    )

    df["NameLen"] = df["Name"].astype(str).str.len()

    df["MissingCount"] = df[[
        "HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin", "Age",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name"
    ]].isna().sum(axis=1)

    # Compact extra features
    df["SpendingPerAge"] = df["Spending"] / (df["Age"].fillna(df["Age"].median()) + 1.0)
    df["Cryo_Age"] = (df["CryoSleep"].astype(str) == "True").astype(int) * df["Age"].fillna(df["Age"].median())
    df["LowSpender"] = (df["Spending"].fillna(0) == 0).astype(int)

    # Group size
    group_counts = df["PassengerId"].astype(str).str.split("_", expand=True)[0].value_counts()
    df["GroupSize"] = df["Group"].map(group_counts).fillna(1).astype(int)

    # Use frequency encoding only if provided
    if freq_maps is not None:
        for col, fmap in freq_maps.items():
            if col in df.columns:
                df[f"{col}_freq"] = df[col].astype(str).map(fmap).fillna(0).astype(float)

    # Normalize categorical missing values
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum", "AgeBin"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Missing").astype(str)

    # Numeric imputation
    num_cols = [
        "Age", "CabinNum", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "Spending", "LogSpending", "NameLen", "MissingCount", "SpendingPerAge",
        "Cryo_Age", "LowSpender", "GroupSize", "AnyLuxury"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    df = df.drop(columns=["Cabin", "Name"], errors="ignore")
    return df


def build_freq_maps(df):
    maps = {}
    freq_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum", "AgeBin"]
    for c in freq_cols:
        if c in df.columns:
            maps[c] = df[c].astype(str).value_counts().to_dict()
    return maps


# Build base features
train_feat = make_features(train)
test_feat = make_features(test)

y = train_feat["Transported"].astype(int)
X = train_feat.drop(columns=["Transported"])

# Build frequency maps from training only (lightweight)
freq_maps = build_freq_maps(train_feat)
train_freq = make_features(train, freq_maps=freq_maps)
test_freq = make_features(test, freq_maps=freq_maps)

X_freq = train_freq.drop(columns=["Transported"])

# Align columns
test = test.copy()
test_base = test_feat[X.columns]
test_freq = test_freq[X_freq.columns]

# Define categorical columns
cat_cols = [c for c in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum", "AgeBin"] if c in X.columns]
cat_cols_freq = [c for c in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "GroupNum", "AgeBin"] if c in X_freq.columns]

# Ensure consistent dtypes
for c in X.columns:
    if c in cat_cols:
        X[c] = X[c].astype(str)
        test_base[c] = test_base[c].astype(str)
    else:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        test_base[c] = pd.to_numeric(test_base[c], errors="coerce")

for c in X_freq.columns:
    if c in cat_cols_freq:
        X_freq[c] = X_freq[c].astype(str)
        test_freq[c] = test_freq[c].astype(str)
    else:
        X_freq[c] = pd.to_numeric(X_freq[c], errors="coerce")
        test_freq[c] = pd.to_numeric(test_freq[c], errors="coerce")

# Small validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Xf_train, Xf_val, _, _ = train_test_split(
    X_freq, y, test_size=0.2, random_state=42, stratify=y
)

cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
cat_idx_freq = [Xf_train.columns.get_loc(c) for c in cat_cols_freq if c in Xf_train.columns]

# -----------------------------
# Faster models: fewer iterations, shallower trees
# -----------------------------
model1 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=450,
    learning_rate=0.06,
    depth=6,
    random_seed=42,
    verbose=0,
    od_type="Iter",
    od_wait=50,
    allow_writing_files=False,
)

model2 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=350,
    learning_rate=0.07,
    depth=5,
    random_seed=42,
    verbose=0,
    od_type="Iter",
    od_wait=40,
    allow_writing_files=False,
)

model3 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=450,
    learning_rate=0.06,
    depth=6,
    random_seed=7,
    verbose=0,
    od_type="Iter",
    od_wait=50,
    allow_writing_files=False,
)

model1.fit(X_train, y_train, cat_features=cat_idx)
val1 = model1.predict_proba(X_val)[:, 1]
test1 = model1.predict_proba(test_base)[:, 1]

model2.fit(Xf_train, y_train, cat_features=cat_idx_freq)
val2 = model2.predict_proba(Xf_val)[:, 1]
test2 = model2.predict_proba(test_freq)[:, 1]

model3.fit(X_train, y_train, cat_features=cat_idx)
val3 = model3.predict_proba(X_val)[:, 1]
test3 = model3.predict_proba(test_base)[:, 1]

val_ens = 0.40 * val1 + 0.35 * val2 + 0.25 * val3
test_ens = 0.40 * test1 + 0.35 * test2 + 0.25 * test3

val_pred = (val_ens >= 0.5).astype(int)
final_validation_score = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Retrain on full data for final submission
full_model1 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=450,
    learning_rate=0.06,
    depth=6,
    random_seed=42,
    verbose=0,
    od_type="Iter",
    od_wait=50,
    allow_writing_files=False,
)
full_model2 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=350,
    learning_rate=0.07,
    depth=5,
    random_seed=42,
    verbose=0,
    od_type="Iter",
    od_wait=40,
    allow_writing_files=False,
)
full_model3 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=450,
    learning_rate=0.06,
    depth=6,
    random_seed=7,
    verbose=0,
    od_type="Iter",
    od_wait=50,
    allow_writing_files=False,
)

full_model1.fit(X, y, cat_features=cat_idx)
full_model2.fit(X_freq, y, cat_features=cat_idx_freq)
full_model3.fit(X, y, cat_features=cat_idx)

full_test1 = full_model1.predict_proba(test_base)[:, 1]
full_test2 = full_model2.predict_proba(test_freq)[:, 1]
full_test3 = full_model3.predict_proba(test_base)[:, 1]

final_test_prob = 0.40 * full_test1 + 0.35 * full_test2 + 0.25 * full_test3
final_test_pred = (final_test_prob >= 0.5)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": final_test_pred.astype(bool)
})
submission.to_csv("submission.csv", index=False)
