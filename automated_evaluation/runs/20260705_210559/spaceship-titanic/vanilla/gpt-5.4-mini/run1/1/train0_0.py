
import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Ensure catboost is available
try:
    from catboost import CatBoostClassifier, Pool
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostClassifier, Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

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
        cabin = df["Cabin"].astype("string")
        parts = cabin.str.split("/", expand=True)
        df["CabinDeck"] = parts[0]
        df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
        df["CabinSide"] = parts[2]
    else:
        df["CabinDeck"] = np.nan
        df["CabinNum"] = np.nan
        df["CabinSide"] = np.nan

    if "Name" in df.columns:
        name = df["Name"].astype("string")
        df["Surname"] = name.str.split().str[-1]
        df["NameLength"] = name.str.len()
    else:
        df["Surname"] = np.nan
        df["NameLength"] = np.nan

    for col in ["CryoSleep", "VIP"]:
        if col in df.columns:
            df[col] = df[col].map(safe_bool_to_int)

    spent_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in df.columns]
    for c in spent_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if spent_cols:
        df["TotalSpent"] = df[spent_cols].sum(axis=1)
        df["NoSpending"] = (df["TotalSpent"] == 0).astype(float)
    else:
        df["TotalSpent"] = np.nan
        df["NoSpending"] = np.nan

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 30, 45, 60, np.inf],
            labels=["child", "teen", "young_adult", "adult", "mid_age", "senior"],
        ).astype("object")
    else:
        df["AgeGroup"] = np.nan

    df = df.drop(columns=["Cabin", "Name"], errors="ignore")

    for col in df.columns:
        if df[col].dtype.name in ["object", "string", "category"]:
            df[col] = df[col].astype("object")

    return df

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y = train_df["Transported"].map(safe_bool_to_int).astype(int)
X = train_df.drop(columns=["Transported"])
test_ids = test_df["PassengerId"].copy()

X_proc = preprocess(X)
test_proc = preprocess(test_df)

feature_cols = [c for c in X_proc.columns if c in test_proc.columns]
X_proc = X_proc[feature_cols].copy()
test_proc = test_proc[feature_cols].copy()

cat_cols = [c for c in X_proc.columns if X_proc[c].dtype == "object"]

for c in feature_cols:
    if c in cat_cols:
        X_proc[c] = X_proc[c].fillna("missing").astype("object")
        test_proc[c] = test_proc[c].fillna("missing").astype("object")
    else:
        X_proc[c] = pd.to_numeric(X_proc[c], errors="coerce")
        test_proc[c] = pd.to_numeric(test_proc[c], errors="coerce")
        med = pd.concat([X_proc[c], test_proc[c]], axis=0).median()
        X_proc[c] = X_proc[c].fillna(med)
        test_proc[c] = test_proc[c].fillna(med)

X_train, X_val, y_train, y_val = train_test_split(
    X_proc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool = Pool(X_val, y_val, cat_features=cat_cols)
test_pool = Pool(test_proc, cat_features=cat_cols)

model = CatBoostClassifier(
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

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

val_pred = (model.predict_proba(val_pool)[:, 1] >= 0.5).astype(int)
final_validation_score = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = (model.predict_proba(test_pool)[:, 1] >= 0.5)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": test_pred.astype(bool)
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to {SUBMISSION_PATH}")
