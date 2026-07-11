
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

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

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_cb = X_train.copy()
X_valid_cb = X_valid.copy()
X_test_cb = X_test.copy()

for c in cat_cols:
    X_train_cb[c] = X_train_cb[c].fillna("Missing")
    X_valid_cb[c] = X_valid_cb[c].fillna("Missing")
    X_test_cb[c] = X_test_cb[c].fillna("Missing")

for c in num_cols:
    med = X_train_cb[c].median()
    X_train_cb[c] = X_train_cb[c].fillna(med)
    X_valid_cb[c] = X_valid_cb[c].fillna(med)
    X_test_cb[c] = X_test_cb[c].fillna(med)

X_train_lgb = X_train.copy()
X_valid_lgb = X_valid.copy()
X_test_lgb = X_test.copy()

for c in X_train_lgb.columns:
    if X_train_lgb[c].dtype == "object" or str(X_train_lgb[c].dtype) == "category":
        X_train_lgb[c] = X_train_lgb[c].fillna("Missing").astype("category")
        X_valid_lgb[c] = X_valid_lgb[c].fillna("Missing").astype("category")
        X_test_lgb[c] = X_test_lgb[c].fillna("Missing").astype("category")
    else:
        med = X_train_lgb[c].median()
        X_train_lgb[c] = X_train_lgb[c].fillna(med)
        X_valid_lgb[c] = X_valid_lgb[c].fillna(med)
        X_test_lgb[c] = X_test_lgb[c].fillna(med)

cat_model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    random_state=42
)

lgb_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42
)

cat_model.fit(
    X_train_cb,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid_cb, y_valid),
    use_best_model=True
)

lgb_model.fit(X_train_lgb, y_train)

valid_pred_cat = cat_model.predict_proba(X_valid_cb)[:, 1]
valid_pred_lgb = lgb_model.predict_proba(X_valid_lgb)[:, 1]
valid_pred = 0.5 * valid_pred_cat + 0.5 * valid_pred_lgb
final_validation_score = roc_auc_score(y_valid, valid_pred)

X_full_cb = X.copy()
X_test_full_cb = X_test.copy()

for c in cat_cols:
    X_full_cb[c] = X_full_cb[c].fillna("Missing")
    X_test_full_cb[c] = X_test_full_cb[c].fillna("Missing")

for c in num_cols:
    med = X_full_cb[c].median()
    X_full_cb[c] = X_full_cb[c].fillna(med)
    X_test_full_cb[c] = X_test_full_cb[c].fillna(med)

X_full_lgb = X.copy()
X_test_full_lgb = X_test.copy()

for c in X_full_lgb.columns:
    if X_full_lgb[c].dtype == "object" or str(X_full_lgb[c].dtype) == "category":
        X_full_lgb[c] = X_full_lgb[c].fillna("Missing").astype("category")
        X_test_full_lgb[c] = X_test_full_lgb[c].fillna("Missing").astype("category")
    else:
        med = X_full_lgb[c].median()
        X_full_lgb[c] = X_full_lgb[c].fillna(med)
        X_test_full_lgb[c] = X_test_full_lgb[c].fillna(med)

final_cat_model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    random_state=42
)

final_lgb_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42
)

final_cat_model.fit(
    X_full_cb,
    y,
    cat_features=cat_cols
)

final_lgb_model.fit(X_full_lgb, y)

test_pred_cat = final_cat_model.predict_proba(X_test_full_cb)[:, 1]
test_pred_lgb = final_lgb_model.predict_proba(X_test_full_lgb)[:, 1]
test_pred = 0.5 * test_pred_cat + 0.5 * test_pred_lgb

submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred
})
submission.to_csv("submission_ensemble.csv", index=False)

print(f"Final Validation Performance: {final_validation_score:.6f}")
