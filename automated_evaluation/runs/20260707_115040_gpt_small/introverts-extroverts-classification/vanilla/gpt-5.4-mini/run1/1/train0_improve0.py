
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

# Identify categorical columns
cat_cols = [c for c in X.columns if X[c].dtype == "object"]

# Fill categorical missing values
for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

# Fill numeric missing values with training medians
num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]
for col in num_cols:
    med = X[col].median()
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# Keep id as numeric
X["id"] = X["id"].fillna(-1)
X_test["id"] = X_test["id"].fillna(-1)

# Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# CatBoost requires categorical feature indices, not names, when using Pool with DataFrame
cat_features_idx = [X.columns.get_loc(c) for c in cat_cols]

train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)
test_pool = Pool(X_test, cat_features=cat_features_idx)

# Model
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=SEED,
    verbose=200,
    early_stopping_rounds=100,
    allow_writing_files=False
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Validation performance
val_pred = model.predict(val_pool)
if isinstance(val_pred, np.ndarray):
    val_pred = val_pred.reshape(-1)
val_pred = val_pred.astype(int)
final_validation_score = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Predict test
test_pred = model.predict(test_pool)
if isinstance(test_pred, np.ndarray):
    test_pred = test_pred.reshape(-1)
test_pred = test_pred.astype(int)

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": [inv_target_map[int(p)] for p in test_pred]
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
