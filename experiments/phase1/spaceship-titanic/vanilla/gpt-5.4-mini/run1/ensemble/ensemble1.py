
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


def preprocess(df, alt=False):
    df = df.copy()

    if "PassengerId" in df.columns:
        df["PassengerId"] = df["PassengerId"].astype(str)

    # Cabin parsing
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

    if alt:
        df["SpendPerPerson"] = df["TotalSpend"] / df["GroupSize"].replace(0, np.nan)
        df["SpendPerPerson"] = df["SpendPerPerson"].replace([np.inf, -np.inf], np.nan)
        df["HasCabin"] = (~df["Deck"].isna()).astype(int)

    # Numerics
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    else:
        df["Age"] = np.nan

    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    numeric_cols = ["Age", "Num"] + spend_cols + ["TotalSpend", "LogTotalSpend", "GroupSize", "NoSpend", "IsAlone"]
    if alt:
        numeric_cols += ["SpendPerPerson", "HasCabin"]

    for c in numeric_cols:
        if c in df.columns:
            if c == "HasCabin":
                df[c] = df[c].fillna(0)
            else:
                df[c] = df[c].fillna(df[c].median())

    # Categorical
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Missing").astype("category")

    drop_cols = ["Cabin", "Name", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def train_and_predict(X, y, test_p, cat_cols, random_state=42, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probs = []
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            iterations=1200,
            learning_rate=0.04,
            depth=6,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            subsample=0.8,   # keep subsampling
            rsm=0.8,
            od_type="Iter",
            od_wait=80
        )

        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_va, y_va),
            use_best_model=True
        )

        val_pred = model.predict(X_va).astype(int).reshape(-1)
        score = accuracy_score(y_va, val_pred)
        print(f"Fold {fold} Validation Performance: {score}")
        scores.append(score)

        probs.append(model.predict_proba(test_p)[:, 1])

    return probs, np.array(scores, dtype=float)


# Preprocess once per variant
train_p_orig = preprocess(train, alt=False)
test_p_orig = preprocess(test, alt=False)
train_p_alt = preprocess(train, alt=True)
test_p_alt = preprocess(test, alt=True)

# Align features
feature_cols_orig = [c for c in train_p_orig.columns if c != "Transported"]
feature_cols_alt = [c for c in train_p_alt.columns if c != "Transported"]

X_orig = train_p_orig[feature_cols_orig].copy()
y_orig = train_p_orig["Transported"].astype(int).copy()
test_p_orig = test_p_orig.reindex(columns=feature_cols_orig, fill_value=np.nan)
cat_cols_orig = [c for c in X_orig.columns if str(X_orig[c].dtype) == "category"]

X_alt = train_p_alt[feature_cols_alt].copy()
y_alt = train_p_alt["Transported"].astype(int).copy()
test_p_alt = test_p_alt.reindex(columns=feature_cols_alt, fill_value=np.nan)
cat_cols_alt = [c for c in X_alt.columns if str(X_alt[c].dtype) == "category"]

# Train only once for each preprocessing variant
probs_orig, scores_orig = train_and_predict(X_orig, y_orig, test_p_orig, cat_cols_orig, random_state=42)
probs_alt, scores_alt = train_and_predict(X_alt, y_alt, test_p_alt, cat_cols_alt, random_state=42)

# Weighted ensemble across the 6 models
all_probas = probs_orig + probs_alt
all_weights = np.concatenate([scores_orig, scores_alt])
if all_weights.sum() == 0:
    all_weights = np.ones_like(all_weights) / len(all_weights)
else:
    all_weights = all_weights / all_weights.sum()

final_test_proba = np.sum([w * p for w, p in zip(all_weights, all_probas)], axis=0)
final_test_pred = final_test_proba >= 0.5

# Simple validation summary
final_validation_score = float(np.mean(np.concatenate([scores_orig, scores_alt])))
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": final_test_pred
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
