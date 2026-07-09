
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


def feature_engineering_solution1(df):
    df = df.copy()

    # Cabin split
    cabin_split = df["Cabin"].fillna("NA/NA/NA").astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0].astype(str)
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].astype(str)

    # Passenger group features
    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0].astype(str)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count").astype(int)

    # Spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

    # Cheap high-signal interactions
    df["AnySpending"] = (df[spending_cols].fillna(0).sum(axis=1) > 0).astype(int)
    df["SpendingPerPerson"] = df["TotalSpending"] / df["GroupSize"].replace(0, np.nan)

    # Optional per-category nonzero indicators
    for col in spending_cols:
        df[f"{col}_NonZero"] = (df[col].fillna(0) > 0).astype(int)

    # Missingness indicators
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]:
        if col in df.columns:
            df[col + "_isna"] = df[col].isna().astype(int)

    # Keep raw cabin/group signals and core categoricals for CatBoost
    categorical_like = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group"]
    for c in categorical_like:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    # Keep booleans as strings to avoid CatBoost categorical type issues
    for c in ["CryoSleep", "VIP"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    return df


def feature_engineering_solution2(df):
    df = df.copy()

    # Cabin split
    cabin_split = df["Cabin"].fillna("NA/NA/NA").astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0].astype(str)
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].astype(str)
    df["CabinKnown"] = (df["Cabin"].notna()).astype(int)
    df["CabinMissingPattern"] = df["Cabin"].isna().astype(int)

    # Passenger group / surname features
    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0].astype(str)
    df["Surname"] = df["Name"].astype(str).fillna("NA").str.split().str[-1].fillna("NA").astype(str)

    # Group structure
    df["HasTicketGroup"] = df["Group"].map(df["Group"].value_counts()).fillna(0).astype(int)
    df["IsAlone"] = (df["Group"].map(df["Group"].value_counts()).fillna(0) == 1).astype(int)

    # Spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["LuxurySpending"] = df[["Spa", "VRDeck"]].sum(axis=1)
    df["BasicSpending"] = df[["RoomService", "FoodCourt", "ShoppingMall"]].sum(axis=1)
    df["AnySpending"] = (df[spending_cols].fillna(0).sum(axis=1) > 0).astype(int)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

    # Spending subgroup indicators
    for col in spending_cols:
        df[f"{col}_NonZero"] = (df[col].fillna(0) > 0).astype(int)

    df["SpendingPerPerson"] = df["TotalSpending"] / df["HasTicketGroup"].replace(0, np.nan)
    df["LuxuryPerBasic"] = df["LuxurySpending"] / df["BasicSpending"].replace(0, np.nan)

    # Age / CabinNum binning
    if "Age" in df.columns:
        df["AgeBin"] = pd.cut(
            pd.to_numeric(df["Age"], errors="coerce"),
            bins=[-1, 12, 18, 30, 50, 80, 200],
            labels=["child", "teen", "young_adult", "adult", "mid_age", "senior"]
        ).astype("string").fillna("NA").astype(str)
        df["Age_isna"] = df["Age"].isna().astype(int)

    df["CabinNumBin"] = pd.cut(
        pd.to_numeric(df["CabinNum"], errors="coerce"),
        bins=[-1, 100, 500, 1000, 2000, 5000, 999999],
        labels=["b1", "b2", "b3", "b4", "b5", "b6"]
    ).astype("string").fillna("NA").astype(str)

    # Missingness indicators
    for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]:
        if col in df.columns:
            df[col + "_isna"] = df[col].isna().astype(int)

    # Categorical handling
    categorical_like = [
        "HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group",
        "Surname", "AgeBin", "CabinNumBin"
    ]
    for c in categorical_like:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    for c in ["CryoSleep", "VIP"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    return df


def prepare_xy(train_df, test_df, feature_fn):
    train_fe = feature_fn(train_df)
    test_fe = feature_fn(test_df)

    y = train_fe["Transported"].astype(int)
    X = train_fe.drop(columns=["Transported"])
    X_test = test_fe.reindex(columns=X.columns, fill_value=np.nan).copy()

    cat_cols = [
        c for c in X.columns
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string")
    ]

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("NA").astype(str)
        X_test[c] = X_test[c].astype("string").fillna("NA").astype(str)

    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        med = X[c].median()
        if pd.isna(med):
            med = 0
        X[c] = X[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

    return X, y, X_test, cat_cols


# Prepare both model datasets
X1, y1, X1_test, cat_cols1 = prepare_xy(train, test, feature_engineering_solution1)
X2, y2, X2_test, cat_cols2 = prepare_xy(train, test, feature_engineering_solution2)

# Same validation split for both models
idx = np.arange(len(train))
idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y1)

X1_train, X1_val = X1.iloc[idx_train], X1.iloc[idx_val]
X2_train, X2_val = X2.iloc[idx_train], X2.iloc[idx_val]
y_train, y_val = y1.iloc[idx_train], y1.iloc[idx_val]

model1 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)

model1.fit(
    X1_train,
    y_train,
    cat_features=cat_cols1,
    eval_set=(X1_val, y_val),
    use_best_model=True
)

val_pred1 = model1.predict(X1_val)
val_acc1 = accuracy_score(y_val, val_pred1)

model2 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)

model2.fit(
    X2_train,
    y_train,
    cat_features=cat_cols2,
    eval_set=(X2_val, y_val),
    use_best_model=True
)

val_pred2 = model2.predict(X2_val)
val_acc2 = accuracy_score(y_val, val_pred2)

# Validation performance of ensemble (for reporting)
val_proba1 = model1.predict_proba(X1_val)[:, 1]
val_proba2 = model2.predict_proba(X2_val)[:, 1]

w1 = val_acc1 / (val_acc1 + val_acc2) if (val_acc1 + val_acc2) > 0 else 0.5
w2 = val_acc2 / (val_acc1 + val_acc2) if (val_acc1 + val_acc2) > 0 else 0.5

val_ensemble_pred = ((w1 * val_proba1 + w2 * val_proba2) >= 0.5).astype(int)
final_validation_score = accuracy_score(y_val, val_ensemble_pred)
print(f"Final Validation Performance: {final_validation_score:.6f}")

# Train final models on full data
final_model1 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=model1.get_best_iteration() if model1.get_best_iteration() is not None else 2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)
final_model1.fit(X1, y1, cat_features=cat_cols1)

final_model2 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=model2.get_best_iteration() if model2.get_best_iteration() is not None else 2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)
final_model2.fit(X2, y2, cat_features=cat_cols2)

test_proba1 = final_model1.predict_proba(X1_test)[:, 1]
test_proba2 = final_model2.predict_proba(X2_test)[:, 1]

test_ensemble_proba = w1 * test_proba1 + w2 * test_proba2
test_pred = (test_ensemble_proba >= 0.5).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)
