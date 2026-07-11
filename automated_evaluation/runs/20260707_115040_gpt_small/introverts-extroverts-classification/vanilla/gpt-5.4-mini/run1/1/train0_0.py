
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

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
val_pool = Pool(X_val, y_val, cat_features=cat_cols)
test_pool = Pool(X_test, cat_features=cat_cols)

model = CatBoostClassifier(
    iterations=1500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=SEED,
    verbose=200
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

val_pred = model.predict(X_val).astype(int).ravel()
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

test_pred = model.predict(test_pool).astype(int).ravel()
submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(test_pred).map(inv_target_map)
})
submission.to_csv("submission.csv", index=False)
