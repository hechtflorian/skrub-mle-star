
import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Ensure catboost is available
from catboost import CatBoostClassifier, Pool

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y = train_df["Transported"].astype(int)
X = train_df.drop(columns=["Transported"])
test_ids = test_df["PassengerId"].copy()

def safe_bool_to_int(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, bool):
        return int(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return 1
    if s in ("false", "0", "no", "n", "f"):
        return 0
    return np.nan

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string").str.split("/", expand=True)
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
        df.drop(columns=["Cabin"], inplace=True)

    if "Name" in df.columns:
        name = df["Name"].astype("string")
        df["Surname"] = name.str.split().str[-1]
        df["NameLength"] = name.str.len()
        df.drop(columns=["Name"], inplace=True)

    for col in ["CryoSleep", "VIP"]:
        if col in df.columns:
            df[col] = df[col].map(safe_bool_to_int)

    spent_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in df.columns]
    for c in spent_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if spent_cols:
        df["TotalSpent"] = df[spent_cols].sum(axis=1)
        df["HasSpending"] = (df["TotalSpent"] > 0).astype(float)
        df["NoSpending"] = (df["TotalSpent"] == 0).astype(float)
    else:
        df["TotalSpent"] = np.nan
        df["HasSpending"] = np.nan
        df["NoSpending"] = np.nan

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 25, 35, 50, 65, np.inf],
            labels=["child", "teen", "young_adult", "adult", "mid_age", "senior", "elder"],
        ).astype("object")
        df["AgeBin"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 25, 35, 50, 65, np.inf],
            labels=False
        )
    else:
        df["AgeGroup"] = np.nan
        df["AgeBin"] = np.nan

    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        df["GroupId"] = pid[0]
        df["GroupMember"] = pd.to_numeric(pid[1], errors="coerce")

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("string"):
            uniq = set(df[col].dropna().astype(str).unique())
            if uniq.issubset({"True", "False"}):
                df[col] = df[col].map({"True": 1, "False": 0})

    return df

X_proc = preprocess(X)
test_proc = preprocess(test_df)

for c in X_proc.columns:
    if c not in test_proc.columns:
        test_proc[c] = np.nan
for c in test_proc.columns:
    if c not in X_proc.columns:
        X_proc[c] = np.nan

feature_cols = [c for c in X_proc.columns if c in test_proc.columns]
X_proc = X_proc[feature_cols].copy()
test_proc = test_proc[feature_cols].copy()

for c in feature_cols:
    if X_proc[c].dtype == "object" or str(X_proc[c].dtype).startswith("string"):
        X_proc[c] = X_proc[c].fillna("missing").astype("object")
        test_proc[c] = test_proc[c].fillna("missing").astype("object")
    else:
        X_proc[c] = pd.to_numeric(X_proc[c], errors="coerce")
        test_proc[c] = pd.to_numeric(test_proc[c], errors="coerce")
        med = pd.concat([X_proc[c], test_proc[c]], axis=0).median()
        X_proc[c] = X_proc[c].fillna(med)
        test_proc[c] = test_proc[c].fillna(med)

cat_cols_cb = [c for c in X_proc.columns if X_proc[c].dtype == "object"]

X_train, X_val, y_train, y_val = train_test_split(
    X_proc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

cat_cols_train = [c for c in X_train.columns if X_train[c].dtype == "object"]

train_pool = Pool(X_train, y_train, cat_features=cat_cols_train)
val_pool = Pool(X_val, y_val, cat_features=cat_cols_train)
test_pool = Pool(test_proc, cat_features=cat_cols_cb)

cat_model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=RANDOM_STATE,
    verbose=False,
    od_type="Iter",
    od_wait=80,
    allow_writing_files=False,
)

cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
cat_val_pred = (cat_model.predict_proba(val_pool)[:, 1] >= 0.5).astype(int)
cat_val_score = accuracy_score(y_val, cat_val_pred)

categorical_cols = [c for c in X_proc.columns if X_proc[c].dtype == "object"]
numeric_cols = [c for c in X_proc.columns if c not in categorical_cols]

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

models = [
    ("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
    ("lr", LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="liblinear",
        random_state=RANDOM_STATE
    )),
    ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE))
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
val_scores = [cat_val_score]
test_probas = [cat_model.predict_proba(test_pool)[:, 1]]

for _, clf in models:
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    cv_score = cross_val_score(pipe, X_proc, y, cv=skf, scoring="accuracy", n_jobs=-1).mean()
    val_scores.append(cv_score)
    pipe.fit(X_proc, y)
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(test_proc)[:, 1]
    else:
        proba = pipe.decision_function(test_proc)
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
    test_probas.append(proba)

weights = np.array(val_scores, dtype=float)
weights = weights / weights.sum()
ensemble_proba = np.zeros(len(test_proc), dtype=float)
for w, p in zip(weights, test_probas):
    ensemble_proba += w * p

val_pred_ensemble = (ensemble_proba >= 0.5).astype(int)
final_validation_score = float(np.mean(val_scores))
print(f"Final Validation Performance: {final_validation_score}")

preds = (ensemble_proba >= 0.5).astype(bool)
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": preds
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to {SUBMISSION_PATH}")
print(submission.head())
