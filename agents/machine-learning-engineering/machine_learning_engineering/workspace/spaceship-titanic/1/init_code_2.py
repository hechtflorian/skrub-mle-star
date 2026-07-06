
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------
# Load data
# -------------------------
INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = train["Transported"].astype(int)
X = train.drop(columns=["Transported"])
test_ids = test["PassengerId"].copy()

# -------------------------
# Feature engineering
# -------------------------
def preprocess(df):
    df = df.copy()

    # Cabin split
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", expand=True)
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
        df.drop(columns=["Cabin"], inplace=True)

    # Name-based family features
    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype(str).str.split().str[-1]
        df["NameLength"] = df["Name"].astype(str).str.len()
        df.drop(columns=["Name"], inplace=True)

    # Spending features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_spend_cols = [c for c in spend_cols if c in df.columns]
    if existing_spend_cols:
        df["TotalSpend"] = df[existing_spend_cols].sum(axis=1)
        df["HasSpending"] = (df["TotalSpend"] > 0).astype(int)
        df["NoSpending"] = (df["TotalSpend"] == 0).astype(int)

    # Age bins
    if "Age" in df.columns:
        df["AgeBin"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 25, 35, 50, 65, np.inf],
            labels=False
        )

    # Passenger group from PassengerId
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        df["GroupId"] = pid[0]
        df["GroupMember"] = pd.to_numeric(pid[1], errors="coerce")

    # Boolean normalization
    for col in df.columns:
        if df[col].dtype == object:
            uniq = set(df[col].dropna().astype(str).unique())
            if uniq.issubset({"True", "False"}):
                df[col] = df[col].map({"True": 1, "False": 0})

    return df

X = preprocess(X)
test = preprocess(test)

# Ensure identical columns
for c in X.columns:
    if c not in test.columns:
        test[c] = np.nan
for c in test.columns:
    if c not in X.columns and c != "Transported":
        X[c] = np.nan

X = X[test.columns]

# -------------------------
# Build model
# -------------------------
categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

models = []

models.append((
    "rf",
    RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
))

models.append((
    "lr",
    LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="liblinear",
        random_state=42
    )
))

models.append((
    "gb",
    GradientBoostingClassifier(random_state=42)
))

# -------------------------
# Cross-validation and ensemble
# -------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
test_probas = []

for name, clf in models:
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    cv_score = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1).mean()
    val_scores.append(cv_score)

    pipe.fit(X, y)
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(test)[:, 1]
    else:
        proba = pipe.decision_function(test)
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
    test_probas.append(proba)

final_validation_score = float(np.mean(val_scores))
print(f"Final Validation Performance: {final_validation_score}")

# Weighted average by validation score
weights = np.array(val_scores)
weights = weights / weights.sum()
ensemble_proba = np.zeros(len(test))
for w, p in zip(weights, test_probas):
    ensemble_proba += w * p

preds = (ensemble_proba >= 0.5).astype(bool)

# -------------------------
# Submission
# -------------------------
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": preds
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
