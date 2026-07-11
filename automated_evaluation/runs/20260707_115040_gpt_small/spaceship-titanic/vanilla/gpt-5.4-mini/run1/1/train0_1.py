
import os
import random
import numpy as np
import pandas as pd
import torch

from catboost import CatBoostClassifier
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

    # Base-solution cabin numeric feature too
    df["CabinNum"] = df["Num"]

    # Group info from PassengerId
    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0].astype(str)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count").astype(int)

    # Surname info from Name
    df["Surname"] = df["Name"].fillna("NA").astype(str).str.split().str[-1].astype(str)
    df["SurnameSize"] = df.groupby("Surname")["PassengerId"].transform("count").astype(int)
    df["HasSurname"] = (df["Surname"] != "NA").astype(int)
    df["SameSurnameGroupSize"] = df.groupby("Surname")["PassengerId"].transform("count").astype(int)

    # Spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)
    df["SpentAny"] = (df["TotalSpending"] > 0).astype(int)

    # Age features
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeBin"] = pd.cut(df["Age"], bins=[-1, 5, 12, 18, 25, 35, 50, 65, 200], labels=False)

    # Missingness indicators
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Num", "CabinNum"]:
        if col in df.columns:
            df[col + "_isna"] = df[col].isna().astype(int)

    # Convert categorical columns to string
    categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "Surname"]
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA")

    return df


train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

y = train_fe["Transported"].astype(int)
X = train_fe.drop(columns=["Transported"]).copy()

# Reindex test to train columns
X_test = test_fe.reindex(columns=X.columns, fill_value=np.nan).copy()

# Drop raw free-text and identifiers that are not directly useful
drop_cols = ["PassengerId", "Cabin", "Name"]
for col in drop_cols:
    if col in X.columns:
        X = X.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

# Detect categorical columns
cat_cols = [
    c for c in X.columns
    if X[c].dtype == "object" or str(X[c].dtype).startswith("string") or str(X[c].dtype) == "bool"
]

# Make sure categorical columns are aligned and string-like for CatBoost, category for LightGBM
for c in cat_cols:
    X[c] = X[c].astype("string").fillna("NA").astype(str)
    X_test[c] = X_test[c].astype("string").fillna("NA").astype(str)

# Numeric columns
num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    med = X[c].median()
    if pd.isna(med):
        med = 0
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

# Prepare LightGBM categorical columns as category dtype with aligned categories
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for c in cat_cols:
    all_cats = pd.Index(
        pd.concat([X_lgb[c].astype("string"), X_test_lgb[c].astype("string")], axis=0)
        .fillna("NA")
        .unique()
    )
    X_lgb[c] = pd.Categorical(X_lgb[c].astype("string").fillna("NA"), categories=all_cats)
    X_test_lgb[c] = pd.Categorical(X_test_lgb[c].astype("string").fillna("NA"), categories=all_cats)

# Hold-out split
X_train_cb, X_val_cb, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train_lgb, X_val_lgb, _, _ = train_test_split(
    X_lgb, y, test_size=0.2, random_state=SEED, stratify=y
)

# CatBoost model
cb_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)
cb_model.fit(
    X_train_cb,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val_cb, y_val),
    use_best_model=True
)
cb_val_pred = cb_model.predict(X_val_cb).astype(int)

# LightGBM model
lgb_model = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)
lgb_model.fit(X_train_lgb, y_train, categorical_feature=cat_cols)
lgb_val_pred = lgb_model.predict(X_val_lgb).astype(int)

# Simple ensemble on validation
cb_val_proba = cb_model.predict_proba(X_val_cb)[:, 1]
lgb_val_proba = lgb_model.predict_proba(X_val_lgb)[:, 1]
ens_val_pred = ((0.5 * cb_val_proba + 0.5 * lgb_val_proba) >= 0.5).astype(int)

cb_acc = accuracy_score(y_val, cb_val_pred)
lgb_acc = accuracy_score(y_val, lgb_val_pred)
ens_acc = accuracy_score(y_val, ens_val_pred)

print(f"Final Validation Performance: {ens_acc:.6f}")

# Train final models on full data
final_cb = CatBoostClassifier(
    loss_function="Logloss",
    iterations=cb_model.get_best_iteration() if cb_model.get_best_iteration() is not None else 2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)
final_cb.fit(X, y, cat_features=cat_cols)

final_lgb = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)
final_lgb.fit(X_lgb, y, categorical_feature=cat_cols)

# Predict test with ensemble
cb_test_proba = final_cb.predict_proba(X_test)[:, 1]
lgb_test_proba = final_lgb.predict_proba(X_test_lgb)[:, 1]
test_pred = ((0.5 * cb_test_proba + 0.5 * lgb_test_proba) >= 0.5).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
