
import os
import random
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = train["Attrition"].astype(int)
X = train.drop(columns=["Attrition"]).copy()
T = test.copy()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

for c in cat_cols:
    X[c] = X[c].astype(str).fillna("NA")
    T[c] = T[c].astype(str).fillna("NA")

num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
    T[c] = T[c].fillna(X[c].median())

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool = Pool(X_val, y_val, cat_features=cat_cols)
test_pool = Pool(T, cat_features=cat_cols)

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=False
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

val_pred = model.predict_proba(val_pool)[:, 1]
final_validation_score = roc_auc_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = model.predict_proba(test_pool)[:, 1]

# Use the correct identifier column from test if available; otherwise fall back to 'id'
id_col = "EmployeeNumber" if "EmployeeNumber" in test.columns else ("id" if "id" in test.columns else None)
if id_col is None:
    submission_ids = np.arange(len(test))
else:
    submission_ids = test[id_col]

submission = pd.DataFrame({
    "EmployeeNumber": submission_ids,
    "Attrition": test_pred
})
submission.to_csv("submission.csv", index=False)
