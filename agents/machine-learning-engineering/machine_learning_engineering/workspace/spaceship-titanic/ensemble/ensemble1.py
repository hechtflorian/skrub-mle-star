
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
from sklearn.metrics import accuracy_score, brier_score_loss

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


def add_spend_features(df, spend_cols):
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


def prepare_data(train_df, test_df):
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

    return X_proc, y, test_proc, test_ids, cat_cols


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_proc, y, test_proc, test_ids, cat_cols = prepare_data(train_df, test_df)

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

# Model A
train_pool_a = Pool(train_df_feat, label=y_train, cat_features=cat_cols)
val_pool_a = Pool(val_df_feat, label=y_val, cat_features=cat_cols)
test_pool_a = Pool(test_df_feat, cat_features=cat_cols)

model_a = CatBoostClassifier(
    iterations=2200,
    learning_rate=0.02,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
)

model_a.fit(train_pool_a)

# Model B: slightly different but still CatBoost, to create complementary error patterns
train_pool_b = Pool(train_df_feat, label=y_train, cat_features=cat_cols)
val_pool_b = Pool(val_df_feat, label=y_val, cat_features=cat_cols)
test_pool_b = Pool(test_df_feat, cat_features=cat_cols)

model_b = CatBoostClassifier(
    iterations=2600,
    learning_rate=0.018,
    depth=7,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=RANDOM_STATE + 7,
    bootstrap_type="Bayesian",
    bagging_temperature=0.35,
    verbose=False,
    allow_writing_files=False,
)

model_b.fit(train_pool_b)

# Validation predictions
val_proba_a = model_a.predict_proba(val_pool_a)[:, 1]
val_proba_b = model_b.predict_proba(val_pool_b)[:, 1]

# Test predictions
test_proba_a = model_a.predict_proba(test_pool_a)[:, 1]
test_proba_b = model_b.predict_proba(test_pool_b)[:, 1]

# Error-aware ensemble plan:
# 3 disagreement bins based on d = abs(p_a - p_b)
val_disagree = np.abs(val_proba_a - val_proba_b)
test_disagree = np.abs(test_proba_a - test_proba_b)

# Bin edges chosen from validation quantiles for robustness
q1, q2 = np.quantile(val_disagree, [0.33, 0.66])
bins = [-np.inf, q1, q2, np.inf]

val_bins = np.digitize(val_disagree, bins[1:-1], right=False)
test_bins = np.digitize(test_disagree, bins[1:-1], right=False)

# Learn fixed weights and thresholds per bin
bin_weights = {}
bin_thresholds = {}
bin_stats = {}

for b in range(3):
    idx = np.where(val_bins == b)[0]
    if len(idx) == 0:
        # fallback
        bin_weights[b] = 0.5
        bin_thresholds[b] = 0.5
        bin_stats[b] = {"acc_a": np.nan, "acc_b": np.nan, "brier_a": np.nan, "brier_b": np.nan}
        continue

    yb = y_val.iloc[idx].values
    pa = val_proba_a[idx]
    pb = val_proba_b[idx]

    # Evaluate simple preference using bin-wise validation performance
    pred_a = (pa >= 0.5).astype(int)
    pred_b = (pb >= 0.5).astype(int)
    acc_a = accuracy_score(yb, pred_a)
    acc_b = accuracy_score(yb, pred_b)
    brier_a = brier_score_loss(yb, pa)
    brier_b = brier_score_loss(yb, pb)

    # Low disagreement: mean
    if b == 0:
        w = 0.5
    else:
        # Medium/high disagreement: weighted toward better performer in that bin
        # Use a small, stable transform from relative error.
        score_a = (acc_a + (1.0 - brier_a)) / 2.0
        score_b = (acc_b + (1.0 - brier_b)) / 2.0
        denom = score_a + score_b
        w = 0.5 if denom <= 0 else float(score_a / denom)
        w = np.clip(w, 0.2, 0.8)

    # Tune threshold on validation bin for the blended probabilities
    blended_val = w * pa + (1.0 - w) * pb
    threshold_grid = np.linspace(0.3, 0.7, 81)
    best_thr = 0.5
    best_score = -1.0
    for thr in threshold_grid:
        pred = (blended_val >= thr).astype(int)
        score = accuracy_score(yb, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    bin_weights[b] = float(w)
    bin_thresholds[b] = best_thr
    bin_stats[b] = {"acc_a": acc_a, "acc_b": acc_b, "brier_a": brier_a, "brier_b": brier_b}

# Apply bin-specific blend/threshold on validation for final metric
val_pred_final = np.zeros_like(y_val.values, dtype=int)
for b in range(3):
    idx = np.where(val_bins == b)[0]
    if len(idx) == 0:
        continue
    w = bin_weights[b]
    thr = bin_thresholds[b]
    blended = w * val_proba_a[idx] + (1.0 - w) * val_proba_b[idx]
    val_pred_final[idx] = (blended >= thr).astype(int)

final_validation_score = accuracy_score(y_val, val_pred_final)
print(f"Final Validation Performance: {final_validation_score}")

# Predict test set using the same bin-specific rules
test_pred_final = np.zeros(len(test_df_feat), dtype=int)
for b in range(3):
    idx = np.where(test_bins == b)[0]
    if len(idx) == 0:
        continue
    w = bin_weights[b]
    thr = bin_thresholds[b]
    blended = w * test_proba_a[idx] + (1.0 - w) * test_proba_b[idx]
    test_pred_final[idx] = (blended >= thr).astype(int)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": test_pred_final.astype(bool)
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to {SUBMISSION_PATH}")
