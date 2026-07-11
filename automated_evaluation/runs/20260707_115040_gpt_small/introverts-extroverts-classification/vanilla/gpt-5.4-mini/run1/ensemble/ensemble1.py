
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

# Keep original columns logic, but create a shared preprocessing pipeline
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]

# Prepare raw copies for feature augmentation before imputation
X_raw = X.copy()
X_test_raw = X_test.copy()

# Fill missing values in categorical columns before creating CatBoost Pool
for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

# Fill numeric missing values with median computed from training data
for col in num_cols:
    med = X[col].median()
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# Keep id as is, but ensure no missing ids (safety)
X["id"] = X["id"].fillna(-1)
X_test["id"] = X_test["id"].fillna(-1)

def add_augmented_features(df_raw, df_imp):
    df = df_imp.copy()

    # Row-level aggregates from original columns before imputation
    raw_num_cols = [c for c in df_raw.columns if c not in cat_cols and c != "id"]

    # Missing count before imputation
    df["row_missing_count"] = df_raw.isna().sum(axis=1).astype(np.int32)

    # Number of categorical columns not equal to "missing"
    if len(cat_cols) > 0:
        cat_not_missing = pd.DataFrame(index=df_raw.index)
        for c in cat_cols:
            cat_not_missing[c] = df_raw[c].notna().astype(np.int32)
        df["cat_not_missing_count"] = cat_not_missing.sum(axis=1).astype(np.int32)
    else:
        df["cat_not_missing_count"] = 0

    # Mean / median of numeric columns per row based on original numeric columns
    if len(raw_num_cols) > 0:
        num_raw = df_raw[raw_num_cols].copy()
        df["row_num_mean"] = num_raw.mean(axis=1)
        df["row_num_median"] = num_raw.median(axis=1)
    else:
        df["row_num_mean"] = 0.0
        df["row_num_median"] = 0.0

    return df

X_aug = add_augmented_features(X_raw, X)
X_test_aug = add_augmented_features(X_test_raw, X_test)

# Identify categorical columns for augmented view
cat_cols_aug = [c for c in X_aug.columns if X_aug[c].dtype == "object"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Same split indices for augmented view
X_aug_tr = X_aug.loc[X_tr.index].copy()
X_aug_val = X_aug.loc[X_val.index].copy()

def make_pools(X_train_df, X_valid_df, X_test_df, cat_cols_local):
    train_pool = Pool(X_train_df, y_tr, cat_features=cat_cols_local)
    val_pool = Pool(X_valid_df, y_val, cat_features=cat_cols_local)
    test_pool = Pool(X_test_df, cat_features=cat_cols_local)
    return train_pool, val_pool, test_pool

def train_model(X_train_df, X_valid_df, X_test_df, cat_cols_local, seed):
    train_pool, val_pool, test_pool = make_pools(X_train_df, X_valid_df, X_test_df, cat_cols_local)
    model = CatBoostClassifier(
        iterations=1500,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=seed,
        verbose=200
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_proba = model.predict_proba(X_valid_df)[:, 1]
    test_proba = model.predict_proba(test_pool)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_pred)
    return model, val_proba, test_proba, val_acc

# Model 1: base view, seed 42
_, val_proba_1, test_proba_1, acc_1 = train_model(X_tr, X_val, X_test, cat_cols, 42)

# Model 2: base view, seed 2024
_, val_proba_2, test_proba_2, acc_2 = train_model(X_tr, X_val, X_test, cat_cols, 2024)

# Model 3: augmented view, seed 42
_, val_proba_3, test_proba_3, acc_3 = train_model(X_aug_tr, X_aug_val, X_test_aug, cat_cols_aug, 42)

# Validation-calibrated weights with soft clipping / minimum floor
scores = np.array([acc_1, acc_2, acc_3], dtype=float)
scores = np.clip(scores, 1e-6, None)
weights = scores / scores.sum()

min_w = 0.15
weights = np.maximum(weights, min_w)
weights = weights / weights.sum()

# Ensemble probabilities
val_ens = weights[0] * val_proba_1 + weights[1] * val_proba_2 + weights[2] * val_proba_3
test_ens = weights[0] * test_proba_1 + weights[1] * test_proba_2 + weights[2] * test_proba_3

# Tune threshold on validation ensemble probabilities
best_thr = 0.5
best_acc = -1.0

coarse_grid = np.arange(0.1, 0.91, 0.02)
for thr in coarse_grid:
    pred = (val_ens >= thr).astype(int)
    acc = accuracy_score(y_val, pred)
    if acc > best_acc:
        best_acc = acc
        best_thr = thr

fine_start = max(0.01, best_thr - 0.03)
fine_end = min(0.99, best_thr + 0.03)
fine_grid = np.arange(fine_start, fine_end + 1e-9, 0.005)
for thr in fine_grid:
    pred = (val_ens >= thr).astype(int)
    acc = accuracy_score(y_val, pred)
    if acc > best_acc:
        best_acc = acc
        best_thr = thr

val_pred_final = (val_ens >= best_thr).astype(int)
final_validation_score = accuracy_score(y_val, val_pred_final)
print(f"Final Validation Performance: {final_validation_score}")

test_pred_final = (test_ens >= best_thr).astype(int)

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(test_pred_final).map(inv_target_map)
})
submission.to_csv("submission.csv", index=False)
