
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

    # Parse Cabin safely
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        df["Deck"] = cabin_split[0]
        df["Num"] = cabin_split[1]
        df["Side"] = cabin_split[2]
    else:
        df["Deck"] = np.nan
        df["Num"] = np.nan
        df["Side"] = np.nan

    # Group features
    if "PassengerId" in df.columns:
        df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
    else:
        df["Group"] = np.nan

    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["GroupSize"] = df["GroupSize"].fillna(1)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # Spending features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))

    # Convert numerics
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    else:
        df["Age"] = np.nan

    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    # Fill missing numeric values
    numeric_cols = ["Age", "Num"] + spend_cols + ["TotalSpend", "LogTotalSpend", "GroupSize"]
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    # Categorical columns
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Missing").astype("category")

    # Drop raw text / ID / leakage-prone columns from features
    drop_cols = ["Cabin", "Name", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

def preprocess_alt(df):
    df = preprocess(df)

    # Add a few safe interaction / binning features
    if "Age" in df.columns:
        age_bins = [-1, 12, 18, 30, 45, 60, 120]
        df["AgeBin"] = pd.cut(df["Age"], bins=age_bins, labels=False, include_lowest=True)
    else:
        df["AgeBin"] = np.nan

    if "TotalSpend" in df.columns and "Age" in df.columns:
        df["SpendPerAge"] = df["TotalSpend"] / (df["Age"].replace(0, np.nan))
    else:
        df["SpendPerAge"] = np.nan

    if "GroupSize" in df.columns and "TotalSpend" in df.columns:
        df["SpendPerGroup"] = df["TotalSpend"] / df["GroupSize"].replace(0, np.nan)
    else:
        df["SpendPerGroup"] = np.nan

    if "HomePlanet" in df.columns and "Destination" in df.columns:
        df["PlanetDest"] = df["HomePlanet"].astype(str) + "_" + df["Destination"].astype(str)
        df["PlanetDest"] = df["PlanetDest"].astype("category")

    if "Deck" in df.columns and "Side" in df.columns:
        df["DeckSide"] = df["Deck"].astype(str) + "_" + df["Side"].astype(str)
        df["DeckSide"] = df["DeckSide"].astype("category")

    # Fill new numeric features
    for c in ["AgeBin", "SpendPerAge", "SpendPerGroup"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    return df

def train_and_predict(train_df, test_df, seed=42, use_alt=False):
    if use_alt:
        train_p = preprocess_alt(train_df)
        test_p = preprocess_alt(test_df)
    else:
        train_p = preprocess(train_df)
        test_p = preprocess(test_df)

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
        random_seed=seed,
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

    va_pred = model.predict(X_va).astype(int).reshape(-1)
    va_score = accuracy_score(y_va, va_pred)

    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter <= 0:
        best_iter = 3000

    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        iterations=best_iter,
        learning_rate=0.03,
        depth=6,
        random_seed=seed,
        verbose=200,
        allow_writing_files=False,
        subsample=0.8,
        rsm=0.8
    )

    final_model.fit(X, y, cat_features=cat_cols)

    test_prob = final_model.predict_proba(test_p)[:, 1]

    return va_score, test_prob

# Model A: original preprocessing + seed 42
score_a, prob_a = train_and_predict(train, test, seed=42, use_alt=False)

# Model B: original preprocessing + different seed
score_b, prob_b = train_and_predict(train, test, seed=2024, use_alt=False)

# Model C: alternative preprocessing + seed 42
score_c, prob_c = train_and_predict(train, test, seed=42, use_alt=True)

# Soft voting / probability averaging
weights = np.array([score_a, score_b, score_c], dtype=float)
if np.any(np.isnan(weights)) or weights.sum() <= 0:
    weights = np.array([1.0, 1.0, 1.0], dtype=float)

weights = weights / weights.sum()
final_prob = weights[0] * prob_a + weights[1] * prob_b + weights[2] * prob_c

# Compute ensemble validation performance using a simple holdout proxy from individual scores
# (the requested metric printed here is the mean validation score from the component models)
final_validation_score = float(np.mean([score_a, score_b, score_c]))
print(f"Final Validation Performance: {final_validation_score}")

test_pred = (final_prob >= 0.5)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
