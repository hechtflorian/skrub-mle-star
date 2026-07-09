
import os
import random
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    # Cabin parsing
    cabin = df["Cabin"].fillna("X/X/X").astype(str).str.split("/", expand=True)
    df["Deck"] = cabin[0].astype(str)
    df["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2].astype(str)

    # Group info from PassengerId
    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0].astype(str)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count").astype(int)

    # Surname info from Name
    df["Surname"] = df["Name"].fillna("NA").astype(str).str.split().str[-1].astype(str)
    df["SurnameSize"] = df.groupby("Surname")["PassengerId"].transform("count").astype(int)

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)
    df["SpentAny"] = (df["TotalSpending"] > 0).astype(int)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeBin"] = pd.cut(df["Age"], bins=[-1, 5, 12, 18, 25, 35, 50, 65, 200], labels=False)

    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Num"]:
        if col in df.columns:
            df[col + "_isna"] = df[col].isna().astype(int)

    # Convert bool-like / categorical columns to string for later category casting
    categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "Surname"]
    for c in categorical_cols:
        df[c] = df[c].astype("string").fillna("NA")

    return df

train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

y = train_fe["Transported"].astype(int)
X = train_fe.drop(columns=["Transported"]).copy()

# Ensure test has same columns as train features
X_test = test_fe.reindex(columns=X.columns, fill_value=np.nan).copy()

# Drop raw free-text and identifiers that are not directly useful as numeric features
# Keep engineered versions (Group, Surname) as categorical features
drop_cols = ["PassengerId", "Cabin", "Name"]
for col in drop_cols:
    if col in X.columns:
        X = X.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

# Detect categorical columns and convert them to pandas 'category' dtype for LightGBM
cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string") or str(X[c].dtype) == "bool"]
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# Numeric columns: coerce to numeric and fill missing values with train medians
num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    med = X[c].median()
    if pd.isna(med):
        med = 0
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

# Align categories between train and test for categorical columns
for c in cat_cols:
    all_cats = pd.Index(pd.concat([X[c].astype("string"), X_test[c].astype("string")], axis=0).fillna("NA").unique())
    X[c] = pd.Categorical(X[c].astype("string").fillna("NA"), categories=all_cats)
    X_test[c] = pd.Categorical(X_test[c].astype("string").fillna("NA"), categories=all_cats)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)

model.fit(X_train, y_train, categorical_feature=cat_cols)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc:.6f}")

final_model = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)

final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict(X_test).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
