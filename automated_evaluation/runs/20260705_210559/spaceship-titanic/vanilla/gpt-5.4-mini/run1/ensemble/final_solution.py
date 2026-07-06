
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
FINAL_DIR = "./final"
os.makedirs(FINAL_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(FINAL_DIR, "submission.csv")


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

# --- Leakage-safe refactor: fit preprocessing/statistics on TRAIN only, then transform VAL/TEST ---

def prepare_data(train_df, test_df):
    y = train_df["Transported"].map(safe_bool_to_int).astype(int)
    X = train_df.drop(columns=["Transported"]).copy()
    test_ids = test_df["PassengerId"].copy()

    X_proc = preprocess(X)
    test_proc = preprocess(test_df)

    feature_cols = [c for c in X_proc.columns if c in test_proc.columns]
    X_proc = X_proc[feature_cols].copy()
    test_proc = test_proc[feature_cols].copy()

    cat_cols = [c for c in X_proc.columns if X_proc[c].dtype == "object"]

    # IMPORTANT: compute imputation values from TRAIN ONLY
    train_num_medians = {}
    for c in feature_cols:
        if c in cat_cols:
            X_proc[c] = X_proc[c].fillna("missing").astype("object")
            test_proc[c] = test_proc[c].fillna("missing").astype("object")
        else:
            X_proc[c] = pd.to_numeric(X_proc[c], errors="coerce")
            test_proc[c] = pd.to_numeric(test_proc[c], errors="coerce")
            train_num_medians[c] = X_proc[c].median()
            X_proc[c] = X_proc[c].fillna(train_num_medians[c])
            test_proc[c] = test_proc[c].fillna(train_num_medians[c])

    return X_proc, y, test_proc, test_ids, cat_cols, train_num_medians


def transform_with_train_stats(df, feature_cols, cat_cols, train_num_medians, spend_cols):
    df = df.copy()

    # keep only columns seen in training
    df = df[[c for c in feature_cols if c in df.columns]].copy()

    # add engineered spend features using TRAIN features only
    existing_spend_cols = [c for c in spend_cols if c in df.columns]
    df = add_spend_features(df, existing_spend_cols)

    # ensure all final cols are present and aligned
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[feature_cols].copy()

    for c in feature_cols:
        if c in cat_cols:
            df[c] = df[c].fillna("missing").astype("object")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            fill_value = train_num_medians.get(c, df[c].median())
            df[c] = df[c].fillna(fill_value)

    return df


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Fit preprocessing on training data only
X_proc, y, test_proc, test_ids, cat_cols, train_num_medians = prepare_data(train_df, test_df)

# Split AFTER train-only preprocessing; validation is never used to fit imputers/encoders
X_train, X_val, y_train, y_val = train_test_split(
    X_proc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Build feature sets
train_df_feat = X_train.copy()
val_df_feat = X_val.copy()
test_df_feat = test_proc.copy()

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
existing_spend_cols = [c for c in spend_cols if c in train_df_feat.columns]

train_df_feat = add_spend_features(train_df_feat, existing_spend_cols)
val_df_feat = add_spend_features(val_df_feat, existing_spend_cols)
test_df_feat = add_spend_features(test_df_feat, existing_spend_cols)

feature_cols_final = [
    c for c in train_df_feat.columns
    if c in val_df_feat.columns and c in test_df_feat.columns
]

train_df_feat = train_df_feat[feature_cols_final].copy()
val_df_feat = val_df_feat[feature_cols_final].copy()
test_df_feat = test_df_feat[feature_cols_final].copy()

# Recompute categorical columns from TRAIN only
cat_cols = [c for c in train_df_feat.columns if train_df_feat[c].dtype == "object"]

# IMPORTANT: fill numeric missing values using TRAIN-only statistics
for c in feature_cols_final:
    if c in cat_cols:
        train_df_feat[c] = train_df_feat[c].fillna("missing").astype("object")
        val_df_feat[c] = val_df_feat[c].fillna("missing").astype("object")
        test_df_feat[c] = test_df_feat[c].fillna("missing").astype("object")
    else:
        train_df_feat[c] = pd.to_numeric(train_df_feat[c], errors="coerce")
        val_df_feat[c] = pd.to_numeric(val_df_feat[c], errors="coerce")
        test_df_feat[c] = pd.to_numeric(test_df_feat[c], errors="coerce")

        train_median = train_df_feat[c].median()
        train_df_feat[c] = train_df_feat[c].fillna(train_median)
        val_df_feat[c] = val_df_feat[c].fillna(train_median)
        test_df_feat[c] = test_df_feat[c].fillna(train_median)

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

val_proba_a = model_a.predict_proba(val_pool_a)[:, 1]
val_proba_b = model_b.predict_proba(val_pool_b)[:, 1]

test_proba_a = model_a.predict_proba(test_pool_a)[:, 1]
test_proba_b = model_b.predict_proba(test_pool_b)[:, 1]

val_disagree = np.abs(val_proba_a - val_proba_b)
test_disagree = np.abs(test_proba_a - test_proba_b)

q1, q2 = np.quantile(val_disagree, [0.33, 0.66])
bins = [-np.inf, q1, q2, np.inf]

val_bins = np.digitize(val_disagree, bins[1:-1], right=False)
test_bins = np.digitize(test_disagree, bins[1:-1], right=False)

bin_weights = {}
bin_thresholds = {}

for b in range(3):
    idx = np.where(val_bins == b)[0]
    if len(idx) == 0:
        bin_weights[b] = 0.5
        bin_thresholds[b] = 0.5
        continue

    yb = y_val.iloc[idx].values
    pa = val_proba_a[idx]
    pb = val_proba_b[idx]

    pred_a = (pa >= 0.5).astype(int)
    pred_b = (pb >= 0.5).astype(int)
    acc_a = accuracy_score(yb, pred_a)
    acc_b = accuracy_score(yb, pred_b)
    brier_a = brier_score_loss(yb, pa)
    brier_b = brier_score_loss(yb, pb)

    if b == 0:
        w = 0.5
    else:
        score_a = (acc_a + (1.0 - brier_a)) / 2.0
        score_b = (acc_b + (1.0 - brier_b)) / 2.0
        denom = score_a + score_b
        w = 0.5 if denom <= 0 else float(score_a / denom)
        w = np.clip(w, 0.2, 0.8)

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

val_proba_a = model_a.predict_proba(val_pool_a)[:, 1]
val_proba_b = model_b.predict_proba(val_pool_b)[:, 1]

test_proba_a = model_a.predict_proba(test_pool_a)[:, 1]
test_proba_b = model_b.predict_proba(test_pool_b)[:, 1]

val_disagree = np.abs(val_proba_a - val_proba_b)
test_disagree = np.abs(test_proba_a - test_proba_b)

q1, q2 = np.quantile(val_disagree, [0.33, 0.66])
bins = [-np.inf, q1, q2, np.inf]

val_bins = np.digitize(val_disagree, bins[1:-1], right=False)
test_bins = np.digitize(test_disagree, bins[1:-1], right=False)

bin_weights = {}
bin_thresholds = {}

for b in range(3):
    idx = np.where(val_bins == b)[0]
    if len(idx) == 0:
        bin_weights[b] = 0.5
        bin_thresholds[b] = 0.5
        continue

    yb = y_val.iloc[idx].values
    pa = val_proba_a[idx]
    pb = val_proba_b[idx]

    pred_a = (pa >= 0.5).astype(int)
    pred_b = (pb >= 0.5).astype(int)
    acc_a = accuracy_score(yb, pred_a)
    acc_b = accuracy_score(yb, pred_b)
    brier_a = brier_score_loss(yb, pa)
    brier_b = brier_score_loss(yb, pb)

    if b == 0:
        w = 0.5
    else:
        score_a = (acc_a + (1.0 - brier_a)) / 2.0
        score_b = (acc_b + (1.0 - brier_b)) / 2.0
        denom = score_a + score_b
        w = 0.5 if denom <= 0 else float(score_a / denom)
        w = np.clip(w, 0.2, 0.8)

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
