
import os
import random
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def feature_engineering(df):
    df = df.copy()

    # Cabin split
    cabin_split = df["Cabin"].fillna("NA/NA/NA").astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0].astype(str)
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].astype(str)

    # PassengerId / Ticket-style structure features
    pid_str = df["PassengerId"].astype(str)
    pid_split = pid_str.str.split("_", expand=True)
    df["Group"] = pid_split[0].astype(str)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count").astype(int)
    df["HasTicketGroup"] = (df["GroupSize"] > 1).astype(int)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # Cabin structure / missingness pattern features
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["CabinMissingPattern"] = np.where(
        df["Cabin"].isna(), "AllMissing", np.where(df["CabinNum"].isna(), "PartialMissing", "Known")
    )

    # Name-related features kept light
    df["Surname"] = df["Name"].fillna("NA").astype(str).str.split().str[-1].astype(str)
    df["HasSurname"] = (df["Surname"] != "NA").astype(int)

    # Spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["TotalSpendingLog1p"] = np.log1p(df["TotalSpending"].clip(lower=0))

    df["EssentialsSpending"] = df[["RoomService", "FoodCourt", "ShoppingMall"]].sum(axis=1)
    df["LuxurySpending"] = df[["Spa", "VRDeck"]].sum(axis=1)
    df["EssentialsSpendingLog1p"] = np.log1p(df["EssentialsSpending"].clip(lower=0))
    df["LuxurySpendingLog1p"] = np.log1p(df["LuxurySpending"].clip(lower=0))

    df["AnySpending"] = (df["TotalSpending"] > 0).astype(int)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)
    df["AnyEssentialsSpending"] = (df["EssentialsSpending"] > 0).astype(int)
    df["AnyLuxurySpending"] = (df["LuxurySpending"] > 0).astype(int)

    # Missingness indicators
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]:
        if col in df.columns:
            df[col + "_isna"] = df[col].isna().astype(int)

    # Discretized numeric features for coarse regime separation
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        if df["Age"].notna().sum() > 1:
            try:
                df["AgeBin"] = pd.qcut(df["Age"], q=5, duplicates="drop").astype(str)
            except Exception:
                df["AgeBin"] = pd.cut(df["Age"], bins=[-np.inf, 12, 18, 30, 45, 60, np.inf], include_lowest=True).astype(str)
        else:
            df["AgeBin"] = "NA"

    if "CabinNum" in df.columns:
        if df["CabinNum"].notna().sum() > 1:
            try:
                df["CabinNumBin"] = pd.qcut(df["CabinNum"], q=5, duplicates="drop").astype(str)
            except Exception:
                df["CabinNumBin"] = pd.cut(
                    df["CabinNum"],
                    bins=[-np.inf, 200, 400, 600, 800, np.inf],
                    include_lowest=True
                ).astype(str)
        else:
            df["CabinNumBin"] = "NA"

    # Force intended categorical columns to string/object dtype
    categorical_like = [
        "HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side",
        "Group", "Surname", "CabinMissingPattern", "AgeBin", "CabinNumBin"
    ]
    for c in categorical_like:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    # Keep booleans as strings to avoid CatBoost categorical type issues
    for c in ["CryoSleep", "VIP"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    return df


train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

y = train_fe["Transported"].astype(int)
X = train_fe.drop(columns=["Transported"])

# Align test to train columns
X_test = test_fe.reindex(columns=X.columns, fill_value=np.nan).copy()

# Identify categorical columns explicitly and ensure they are string-like
cat_cols = [
    c for c in X.columns
    if X[c].dtype == "object" or str(X[c].dtype).startswith("string")
]

# Convert categorical columns in both train/test to string with a single missing token
for c in cat_cols:
    X[c] = X[c].astype("string").fillna("NA").astype(str)
    X_test[c] = X_test[c].astype("string").fillna("NA").astype(str)

# Fill numeric missing values
num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    med = X[c].median()
    if pd.isna(med):
        med = 0
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

# Hold-out validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    use_best_model=True
)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc:.6f}")

# Train on full data and predict test set
final_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=model.get_best_iteration() if model.get_best_iteration() is not None else 2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)

final_model.fit(X, y, cat_features=cat_cols)

test_pred = final_model.predict(X_test).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)
