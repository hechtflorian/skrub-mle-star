
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


def add_spend_features(df: pd.DataFrame, spend_cols):
    df = df.copy()
    if spend_cols:
        spend_sum = df[spend_cols].sum(axis=1)
    else:
        spend_sum = pd.Series(0, index=df.index, dtype=float)

    df["TotalSpend"] = spend_sum
    df["AnySpend"] = (spend_sum > 0).astype(int)
    df["ZeroSpend"] = (spend_sum == 0).astype(int)
    df["LogTotalSpend"] = np.log1p(spend_sum)

    if "PassengerGroupSize" in df.columns:
        group_size = pd.to_numeric(df["PassengerGroupSize"], errors="coerce").fillna(1)
        group_size = group_size.replace(0, 1)
        df["SpendPerPerson"] = spend_sum / group_size
    else:
        df["SpendPerPerson"] = spend_sum

    df["LogSpendPerPerson"] = np.log1p(df["SpendPerPerson"].clip(lower=0))

    if "CryoSleep" in df.columns:
        cryo = pd.to_numeric(df["CryoSleep"], errors="coerce").fillna(0)
        df["LogAvgSpend"] = np.log1p(spend_sum / (cryo + 1))
    else:
        df["LogAvgSpend"] = np.log1p(spend_sum / 2.0)

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

# Rebuild dataframes for feature engineering, keeping labels aligned
train_df_feat = X_train.copy()
val_df_feat = X_val.copy()
test_df_feat = test_proc.copy()

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
existing_spend_cols = [c for c in spend_cols if c in train_df_feat.columns]

train_df_feat = add_spend_features(train_df_feat, existing_spend_cols)
val_df_feat = add_spend_features(val_df_feat, existing_spend_cols)
test_df_feat = add_spend_features(test_df_feat, existing_spend_cols)

# Ensure exact same columns across splits
feature_cols_final = [c for c in train_df_feat.columns if c in val_df_feat.columns and c in test_df_feat.columns]
train_df_feat = train_df_feat[feature_cols_final].copy()
val_df_feat = val_df_feat[feature_cols_final].copy()
test_df_feat = test_df_feat[feature_cols_final].copy()

cat_cols = [c for c in train_df_feat.columns if train_df_feat[c].dtype == "object"]

for c in feature_cols_final:
    if c in cat_cols:
        train_df_feat[c] = train_df_feat[c].fillna("missing").astype("object")
        val_df_feat[c] = val_df_feat[c].fillna("missing").astype("object")
        test_df_feat[c] = test_df_feat[c].fillna("missing").astype("object")
    else:
        train_df_feat[c] = pd.to_numeric(train_df_feat[c], errors="coerce")
        val_df_feat[c] = pd.to_numeric(val_df_feat[c], errors="coerce")
        test_df_feat[c] = pd.to_numeric(test_df_feat[c], errors="coerce")
        med = pd.concat([train_df_feat[c], val_df_feat[c], test_df_feat[c]], axis=0).median()
        train_df_feat[c] = train_df_feat[c].fillna(med)
        val_df_feat[c] = val_df_feat[c].fillna(med)
        test_df_feat[c] = test_df_feat[c].fillna(med)

train_pool = Pool(train_df_feat, label=y_train, cat_features=cat_cols)
val_pool = Pool(val_df_feat, label=y_val, cat_features=cat_cols)
test_pool = Pool(test_df_feat, cat_features=cat_cols)

model = CatBoostClassifier(
    iterations=2200,
    learning_rate=0.02,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
)

model.fit(train_pool)

val_proba = model.predict_proba(val_pool)[:, 1]

threshold_grid = np.linspace(0.3, 0.7, 81)
best_threshold = 0.5
best_val_score = -1.0

for thr in threshold_grid:
    val_pred_thr = (val_proba >= thr).astype(int)
    thr_score = accuracy_score(y_val, val_pred_thr)
    if thr_score > best_val_score:
        best_val_score = thr_score
        best_threshold = thr

final_validation_score = best_val_score
print(f"Final Validation Performance: {final_validation_score}")
print(f"Best Threshold: {best_threshold}")

test_proba = model.predict_proba(test_pool)[:, 1]
test_pred = (test_proba >= best_threshold)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": test_pred.astype(bool)
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to {SUBMISSION_PATH}")
