
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_map = {"Introvert": 0, "Extrovert": 1}
inv_target_map = {0: "Introvert", 1: "Extrovert"}

y = train["Personality"].map(target_map).astype(int)
X = train.drop(columns=["Personality"]).copy()
X_test = test.copy()

# Fill missing values in categorical columns before creating CatBoost Pool
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

# Fill numeric missing values with median computed from training data
num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]
for col in num_cols:
    med = X[col].median()
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# Keep id as is, but ensure no missing ids (safety)
X["id"] = X["id"].fillna(-1)
X_test["id"] = X_test["id"].fillna(-1)

# Base split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Build a second feature view with missing indicators
def add_missing_indicators(df, original_df):
    df2 = df.copy()
    for col in original_df.columns:
        if col == "id" or col == "Personality":
            continue
        df2[f"{col}_was_missing"] = original_df[col].isna().astype(int).values if len(original_df) == len(df) else 0
    return df2

# For full train/test, create indicator features from the pre-imputation raw data
raw_train = train.drop(columns=["Personality"]).copy()
raw_test = test.copy()

indicator_train = pd.DataFrame(index=raw_train.index)
indicator_test = pd.DataFrame(index=raw_test.index)
for col in raw_train.columns:
    if col == "id":
        continue
    indicator_train[f"{col}_was_missing"] = raw_train[col].isna().astype(int)
    indicator_test[f"{col}_was_missing"] = raw_test[col].isna().astype(int)

X_ind = pd.concat([X.reset_index(drop=True), indicator_train.reset_index(drop=True)], axis=1)
X_test_ind = pd.concat([X_test.reset_index(drop=True), indicator_test.reset_index(drop=True)], axis=1)

# Re-split indicator view using same indices
X_ind_tr = X_ind.loc[X_tr.index].reset_index(drop=True)
X_ind_val = X_ind.loc[X_val.index].reset_index(drop=True)
X_test_ind = X_test_ind.reset_index(drop=True)

# Base view reset indexes
X_tr = X_tr.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_tr = y_tr.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# Detect categorical columns for each view
cat_cols_base = [c for c in X_tr.columns if X_tr[c].dtype == "object"]
cat_cols_ind = [c for c in X_ind_tr.columns if X_ind_tr[c].dtype == "object"]

# Variants: base and missing-indicator versions with different seeds
model_configs = [
    {
        "name": "base_seed42",
        "X_tr": X_tr,
        "X_val": X_val,
        "X_test": X_test,
        "cat_cols": cat_cols_base,
        "seed": 42,
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 1500,
    },
    {
        "name": "base_seed2024",
        "X_tr": X_tr,
        "X_val": X_val,
        "X_test": X_test,
        "cat_cols": cat_cols_base,
        "seed": 2024,
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 1500,
    },
    {
        "name": "missing_seed42",
        "X_tr": X_ind_tr,
        "X_val": X_ind_val,
        "X_test": X_test_ind,
        "cat_cols": cat_cols_ind,
        "seed": 42,
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 1500,
    },
]

val_probas = []
test_probas = []

for cfg in model_configs:
    train_pool = Pool(cfg["X_tr"], y_tr, cat_features=cfg["cat_cols"])
    val_pool = Pool(cfg["X_val"], y_val, cat_features=cfg["cat_cols"])
    test_pool = Pool(cfg["X_test"], cat_features=cfg["cat_cols"])

    model = CatBoostClassifier(
        iterations=cfg["iterations"],
        depth=cfg["depth"],
        learning_rate=cfg["learning_rate"],
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=cfg["seed"],
        verbose=200,
        allow_writing_files=False
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred_proba = model.predict_proba(cfg["X_val"])[:, 1]
    test_pred_proba = model.predict_proba(cfg["X_test"])[:, 1]

    val_probas.append(val_pred_proba)
    test_probas.append(test_pred_proba)

# Average probabilities across models
val_ensemble_proba = np.mean(np.vstack(val_probas), axis=0)
test_ensemble_proba = np.mean(np.vstack(test_probas), axis=0)

# Tune a global threshold on validation set
thresholds = np.linspace(0.1, 0.9, 801)
best_threshold = 0.5
best_acc = -1

for thr in thresholds:
    val_pred = (val_ensemble_proba >= thr).astype(int)
    acc = accuracy_score(y_val, val_pred)
    if acc > best_acc:
        best_acc = acc
        best_threshold = thr

print(f"Final Validation Performance: {best_acc}")

test_pred = (test_ensemble_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(test_pred).map(inv_target_map)
})
submission.to_csv("submission.csv", index=False)
