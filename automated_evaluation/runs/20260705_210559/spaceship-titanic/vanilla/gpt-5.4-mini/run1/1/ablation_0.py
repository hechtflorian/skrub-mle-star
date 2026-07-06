
import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

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

def preprocess(df: pd.DataFrame, use_name_features=True, use_cabin_features=True, use_spending_features=True) -> pd.DataFrame:
    df = df.copy()

    if use_cabin_features and "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string")
        parts = cabin.str.split("/", expand=True)
        df["CabinDeck"] = parts[0]
        df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
        df["CabinSide"] = parts[2]
    else:
        df["CabinDeck"] = np.nan
        df["CabinNum"] = np.nan
        df["CabinSide"] = np.nan

    if use_name_features and "Name" in df.columns:
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
    if not use_spending_features:
        spent_cols = []

    for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if not use_spending_features:
                df[c] = np.nan

    if use_spending_features and spent_cols:
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

def prepare_data(train_df, ablation_name, use_name_features=True, use_cabin_features=True, use_spending_features=True, use_od=True, use_age_group=True):
    y = train_df["Transported"].map(safe_bool_to_int).astype(int)
    X = train_df.drop(columns=["Transported"])

    
def prepare_data(train_df, ablation_name, use_name_features=True, use_cabin_features=True, use_spending_features=True, use_od=True, use_age_group=True):
    y = train_df["Transported"].map(safe_bool_to_int).astype(int)
    X = train_df.drop(columns=["Transported"])

    # Split first so any preprocessing/statistics are learned ONLY from the training split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocess train/validation independently to avoid leakage from validation into training
    X_train = preprocess(
        X_train_raw,
        use_name_features=use_name_features,
        use_cabin_features=use_cabin_features,
        use_spending_features=use_spending_features,
    )
    X_val = preprocess(
        X_val_raw,
        use_name_features=use_name_features,
        use_cabin_features=use_cabin_features,
        use_spending_features=use_spending_features,
    )

    if not use_age_group and "AgeGroup" in X_train.columns:
        X_train["AgeGroup"] = np.nan
    if not use_age_group and "AgeGroup" in X_val.columns:
        X_val["AgeGroup"] = np.nan

    # Ensure both splits have identical columns
    X_train, X_val = X_train.align(X_val, join="outer", axis=1)

    # Learn imputations and encodings using only the training split
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

    for c in X_train.columns:
        if c in cat_cols:
            X_train[c] = X_train[c].fillna("missing").astype("object")
            X_val[c] = X_val[c].fillna("missing").astype("object")
        else:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_val[c] = pd.to_numeric(X_val[c], errors="coerce")
            med = X_train[c].median()
            X_train[c] = X_train[c].fillna(med)
            X_val[c] = X_val[c].fillna(med)

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter" if use_od else None,
        od_wait=80 if use_od else None,
    )

    # Train ONLY on training samples; validation is used only for evaluation / early stopping
    model.fit(train_pool, eval_set=val_pool, use_best_model=use_od)

    # Final validation score is computed on held-out validation data only
    val_pred = (model.predict_proba(val_pool)[:, 1] >= 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred)

    print(f"{ablation_name}: validation accuracy = {acc:.5f}")
    return acc

    acc = accuracy_score(y_val, val_pred)

    print(f"{ablation_name}: validation accuracy = {acc:.5f}")
    return acc

train_df = pd.read_csv(TRAIN_PATH)

results = {}

# Baseline
results["baseline"] = prepare_data(
    train_df,
    "Baseline",
    use_name_features=True,
    use_cabin_features=True,
    use_spending_features=True,
    use_od=True,
)

# Ablation 1: remove Cabin-derived features
results["no_cabin"] = prepare_data(
    train_df,
    "Ablation 1 - No Cabin Features",
    use_name_features=True,
    use_cabin_features=False,
    use_spending_features=True,
    use_od=True,
)

# Ablation 2: remove Name-derived features
results["no_name"] = prepare_data(
    train_df,
    "Ablation 2 - No Name Features",
    use_name_features=False,
    use_cabin_features=True,
    use_spending_features=True,
    use_od=True,
)

# Ablation 3: remove spending features
results["no_spending"] = prepare_data(
    train_df,
    "Ablation 3 - No Spending Features",
    use_name_features=True,
    use_cabin_features=True,
    use_spending_features=False,
    use_od=True,
)

# Ablation 4: disable early stopping
results["no_od"] = prepare_data(
    train_df,
    "Ablation 4 - No Early Stopping",
    use_name_features=True,
    use_cabin_features=True,
    use_spending_features=True,
    use_od=False,
)

baseline = results["baseline"]
deltas = {k: v - baseline for k, v in results.items() if k != "baseline"}

print("\nPerformance deltas vs baseline:")
for k, delta in deltas.items():
    print(f"{k}: {delta:+.5f}")

most_important = min(deltas, key=lambda k: deltas[k])  # biggest drop when removed
print(
    f"\nMost important part for performance: {most_important} "
    f"(largest accuracy drop of {deltas[most_important]:+.5f} when ablated)"
)
