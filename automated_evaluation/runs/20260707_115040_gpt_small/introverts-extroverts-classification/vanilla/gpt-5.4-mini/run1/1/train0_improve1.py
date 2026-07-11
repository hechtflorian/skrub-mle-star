
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

# Prepare target and features
y = train["Personality"].map(target_map).astype(int)
X = train.drop(columns=["Personality"]).copy()
X_test = test.copy()

# Identify categorical and numeric columns
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]

# Handle missing values before split for consistency
for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

for col in num_cols:
    med = X[col].median()
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# Ensure id is usable
X["id"] = X["id"].fillna(-1)
X_test["id"] = X_test["id"].fillna(-1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Feature engineering should use the split data that actually exists
X_train_fe = X_train.copy()
X_val_fe = X_val.copy()
test_fe = X_test.copy()

numeric_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    miss_col = f"{col}_is_missing"
    X_train_fe[miss_col] = X_train_fe[col].isna().astype(np.int8)
    X_val_fe[miss_col] = X_val_fe[col].isna().astype(np.int8)
    test_fe[miss_col] = test_fe[col].isna().astype(np.int8)

X_train_fe["missing_count"] = X_train_fe[numeric_cols].isna().sum(axis=1).astype(np.int16)
X_val_fe["missing_count"] = X_val_fe[numeric_cols].isna().sum(axis=1).astype(np.int16)
test_fe["missing_count"] = test_fe[numeric_cols].isna().sum(axis=1).astype(np.int16)

# CatBoost categorical features must match the engineered frames
cat_features = [c for c in X_train_fe.columns if X_train_fe[c].dtype == "object"]

train_pool_fe = Pool(X_train_fe, y_train, cat_features=cat_features)
val_pool_fe = Pool(X_val_fe, y_val, cat_features=cat_features)
test_pool_fe = Pool(test_fe, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=4000,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=200,
    l2_leaf_reg=8.0,
    random_strength=1.0,
    od_type="Iter",
    od_wait=200,
    allow_writing_files=False
)

model.fit(train_pool_fe, eval_set=val_pool_fe, use_best_model=True)

# Validation threshold tuning
val_proba = model.predict_proba(val_pool_fe)[:, 1]
thresholds = np.linspace(0.05, 0.95, 181)
val_accs = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    val_accs.append(accuracy_score(y_val, pred))

best_idx = int(np.argmax(val_accs))
best_threshold = float(thresholds[best_idx])
final_validation_score = float(val_accs[best_idx])

print(f"Best validation threshold: {best_threshold:.4f}")
print(f"Final Validation Performance: {final_validation_score}")

# Refit on full data using the same feature engineering
full_X = pd.concat([X_train_fe, X_val_fe], axis=0).reset_index(drop=True)
full_y = pd.concat([pd.Series(y_train).reset_index(drop=True), pd.Series(y_val).reset_index(drop=True)], axis=0).reset_index(drop=True)

full_pool = Pool(full_X, full_y, cat_features=cat_features)

best_iter = model.get_best_iteration()
if best_iter is None or best_iter <= 0:
    best_iter = 1000

final_model = CatBoostClassifier(
    iterations=best_iter,
    depth=6,
    learning_rate=0.03,
    loss_function="Loglog",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=200,
    l2_leaf_reg=8.0,
    random_strength=1.0,
    allow_writing_files=False
)

# Fix typo if model param is invalid by ensuring proper loss function
final_model = CatBoostClassifier(
    iterations=best_iter,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=200,
    l2_leaf_reg=8.0,
    random_strength=1.0,
    allow_writing_files=False
)

final_model.fit(full_pool, verbose=200)

test_proba = final_model.predict_proba(test_pool_fe)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(test_pred).map(inv_target_map)
})

submission.to_csv("submission.csv", index=False)
