
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

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

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

for c in cat_cols:
    X[c] = X[c].fillna("Missing")
    X_test[c] = X_test[c].fillna("Missing")

for c in num_cols:
    med = X[c].median()
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

valid_pred = model.predict_proba(X_valid)[:, 1]
final_validation_score = roc_auc_score(y_valid, valid_pred)

test_pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred
})
submission.to_csv("submission_catboost.csv", index=False)

print(f"Final Validation Performance: {final_validation_score:.6f}")
