
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

for c in cat_cols:
    if X[c].dtype == "object":
        X[c] = X[c].astype("string")
    if X_test[c].dtype == "object":
        X_test[c] = X_test[c].astype("string")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

search_space = [
    {"depth": 5, "l2_leaf_reg": 3, "iterations": 1000},
    {"depth": 6, "l2_leaf_reg": 3, "iterations": 1200},
    {"depth": 6, "l2_leaf_reg": 5, "iterations": 1400},
    {"depth": 7, "l2_leaf_reg": 5, "iterations": 1200},
    {"depth": 7, "l2_leaf_reg": 7, "iterations": 1400},
]

best_score = -1
best_model = None
best_params = None

for params in search_space:
    candidate_model = CatBoostClassifier(
        iterations=params["iterations"],
        learning_rate=0.03,
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        verbose=200
    )

    candidate_model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=100
    )

    valid_pred = candidate_model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)

    if score > best_score:
        best_score = score
        best_model = candidate_model
        best_params = params

model = best_model
final_validation_score = best_score
valid_pred = model.predict_proba(X_valid)[:, 1]
test_pred = model.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred
})
submission.to_csv("submission_catboost.csv", index=False)

print(f"Final Validation Performance: {final_validation_score:.6f}")
