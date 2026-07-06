
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

    # Cabin parsing
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

    # Name parsing
    if "Name" in df.columns:
        name = df["Name"].astype("string")
        df["Surname"] = name.str.split().str[-1]
        df["NameLength"] = name.str.len()
    else:
        df["Surname"] = np.nan
        df["NameLength"] = np.nan

    # Boolean features
    for col in ["CryoSleep", "VIP"]:
        if col in df.columns:
            df[col] = df[col].map(safe_bool_to_int)

    # Spending features
    spent_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in df.columns]
    for c in spent_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if spent_cols:
        df["TotalSpent"] = df[spent_cols].sum(axis=1)
        df["NoSpending"] = (df["TotalSpent"] == 0).astype(float)
        df["SpentVar"] = df[spent_cols].var(axis=1)
    else:
        df["TotalSpent"] = np.nan
        df["NoSpending"] = np.nan
        df["SpentVar"] = np.nan

    # Age features
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 30, 45, 60, np.inf],
            labels=["child", "teen", "young_adult", "adult", "mid_age", "senior"],
        ).astype("object")
        df["IsChild"] = (df["Age"] <= 12).astype(float)
        df["IsSenior"] = (df["Age"] >= 60).astype(float)
    else:
        df["AgeGroup"] = np.nan
        df["IsChild"] = np.nan
        df["IsSenior"] = np.nan

    # Drop raw text columns
    df = df.drop(columns=["Cabin", "Name"], errors="ignore")

    # Ensure categorical-like columns remain object for CatBoost
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

# Align columns between train and test
feature_cols = [c for c in X_proc.columns if c in test_proc.columns]
X_proc = X_proc[feature_cols].copy()
test_proc = test_proc[feature_cols].copy()

# Identify categorical columns before imputation
cat_cols = [c for c in X_proc.columns if X_proc[c].dtype == "object"]

# Fill missing values
for c in feature_cols:
    if c in cat_cols:
        X_proc[c] = X_proc[c].fillna("missing").astype("object")
        test_proc[c] = test_proc[c].fillna("missing").astype("object")
    else:
        X_proc[c] = pd.to_numeric(X_proc[c], errors="coerce")
        test_proc[c] = pd.to_numeric(test_proc[c], errors="coerce")
        med = pd.concat([X_proc[c], test_proc[c]], axis=0).median()
        if pd.isna(med):
            med = 0.0
        X_proc[c] = X_proc[c].fillna(med)
        test_proc[c] = test_proc[c].fillna(med)

# Add simple interaction features that are commonly useful
for df in (X_proc, test_proc):
    if all(c in df.columns for c in ["Age", "TotalSpent"]):
        df["Age_x_Spent"] = df["Age"] * df["TotalSpent"]
    if all(c in df.columns for c in ["CryoSleep", "TotalSpent"]):
        df["CryoSleep_x_Spent"] = df["CryoSleep"] * df["TotalSpent"]

# Recalculate categorical columns after new features
cat_cols = [c for c in X_proc.columns if X_proc[c].dtype == "object"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_proc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool = Pool(X_val, y_val, cat_features=cat_cols)
test_pool = Pool(test_proc, cat_features=cat_cols)

# Model
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=2500,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    verbose=200,
    early_stopping_rounds=150,
    allow_writing_files=False,
    auto_class_weights="Balanced",
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Validation performance
val_pred_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_pred_proba >= 0.5).astype(int)
final_validation_score = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full data for final submission
full_pool = Pool(X_proc, y, cat_features=cat_cols)
final_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=int(model.get_best_iteration() or 2500),
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    verbose=200,
    allow_writing_files=False,
    auto_class_weights="Balanced",
)
final_model.fit(full_pool)

test_pred_proba = final_model.predict_proba(test_proc)[:, 1]
test_pred = test_pred_proba >= 0.5

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": test_pred.astype(bool)
})

submission.to_csv(SUBMISSION_PATH, index=False)
