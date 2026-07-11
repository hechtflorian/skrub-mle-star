
import os
import random
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

train_path = os.path.join(".", "input", "train.csv")
test_path = os.path.join(".", "input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "Attrition"
id_col = "id" if "id" in test.columns else "EmployeeNumber"

X = train.drop(columns=[target]).copy()
y = train[target].copy()
X_test = test.copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

for c in X_train.columns:
    if X_train[c].dtype == "object":
        X_train[c] = X_train[c].fillna("Missing").astype("category")
        X_valid[c] = X_valid[c].fillna("Missing").astype("category")
        X_test[c] = X_test[c].fillna("Missing").astype("category")
    else:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_valid[c] = X_valid[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42
)

model.fit(X_train, y_train)

valid_pred = model.predict_proba(X_valid)[:, 1]
final_validation_score = roc_auc_score(y_valid, valid_pred)

X_full = X.copy()
X_test_full = test.copy()

for c in X_full.columns:
    if X_full[c].dtype == "object":
        X_full[c] = X_full[c].fillna("Missing").astype("category")
        X_test_full[c] = X_test_full[c].fillna("Missing").astype("category")
    else:
        med = X_full[c].median()
        X_full[c] = X_full[c].fillna(med)
        X_test_full[c] = X_test_full[c].fillna(med)

final_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42
)

final_model.fit(X_full, y)

test_pred = final_model.predict_proba(X_test_full)[:, 1]

submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred
})
submission.to_csv("submission_lgbm.csv", index=False)

print(f"Final Validation Performance: {final_validation_score:.6f}")
