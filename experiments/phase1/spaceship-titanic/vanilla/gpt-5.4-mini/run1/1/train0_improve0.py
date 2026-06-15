
import sys
import subprocess
import importlib.util
import warnings

warnings.filterwarnings("ignore")

# Install catboost if missing
if importlib.util.find_spec("catboost") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


def preprocess(df):
    df = df.copy()

    # Keep PassengerId only for later output; do not use as a feature
    if "PassengerId" in df.columns:
        df["PassengerId"] = df["PassengerId"].astype(str)

    # Parse Cabin safely and add missingness indicators
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        df["Deck"] = cabin_split[0]
        df["Num"] = cabin_split[1]
        df["Side"] = cabin_split[2]
    else:
        df["Deck"] = np.nan
        df["Num"] = np.nan
        df["Side"] = np.nan

    df["DeckMissing"] = df["Deck"].isna().astype(int)
    df["SideMissing"] = df["Side"].isna().astype(int)

    # Treat cabin number as ordinal numeric
    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    # Simplify/remove group features per validation signal
    if "PassengerId" in df.columns:
        df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
    else:
        df["Group"] = np.nan

    # Spending features: robust transforms, zero flags, interactions
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

        df[f"{c}_is_zero"] = (df[c].fillna(0) == 0).astype(int)
        df[f"{c}_log1p"] = np.log1p(df[c].fillna(0))

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))
    df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)
    df["SpendCount"] = df[spend_cols].fillna(0).gt(0).sum(axis=1)
    df["SpendDiversity"] = df["SpendCount"] / len(spend_cols)

    # Convert numerics
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    else:
        df["Age"] = np.nan

    # Fill missing numeric values
    numeric_cols = (
        ["Age", "Num", "TotalSpend", "LogTotalSpend", "SpendCount", "SpendDiversity"]
        + spend_cols
        + [f"{c}_log1p" for c in spend_cols]
    )
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Categorical columns
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Missing").astype("category")

    # Drop raw text / ID / leakage-prone columns from features
    drop_cols = ["Cabin", "Name", "PassengerId", "Group"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

train_p = preprocess(train)
test_p = preprocess(test)

target = "Transported"
feature_cols = [c for c in train_p.columns if c != target]

X = train_p[feature_cols].copy()
y = train_p[target].astype(int).copy()

# Align test columns to train columns
test_p = test_p.reindex(columns=feature_cols, fill_value=np.nan)

# Ensure categorical columns are correctly identified
cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    subsample=0.8,
    rsm=0.8
)

model.fit(
    X_tr,
    y_tr,
    cat_features=cat_cols,
    eval_set=(X_va, y_va),
    use_best_model=True
)

val_pred = model.predict(X_va).astype(int).reshape(-1)
final_validation_score = accuracy_score(y_va, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

best_iter = model.get_best_iteration()
if best_iter is None or best_iter <= 0:
    best_iter = 3000

final_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=best_iter,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    subsample=0.8,
    rsm=0.8
)

final_model.fit(X, y, cat_features=cat_cols)

test_pred = final_model.predict(test_p).astype(bool).reshape(-1)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
