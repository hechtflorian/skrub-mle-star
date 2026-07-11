
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "booking_status"
id_col = "id"

X = train_df.drop(columns=[target_col])
y = train_df[target_col].astype(int)
X_test = test_df.copy()

# Identify categorical columns, including any object columns and low-cardinality integer columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for col in X.columns:
    if col != id_col and X[col].nunique(dropna=False) <= 20 and col not in cat_cols:
        cat_cols.append(col)

# Remove id from features
X = X.drop(columns=[id_col])
X_test = X_test.drop(columns=[id_col])

# Train/validation split
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cast categorical columns
for c in cat_cols:
    if c in X_tr.columns:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category")
        X_test[c] = X_test[c].astype("category")

# LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]
)

va_pred = model.predict_proba(X_va)[:, 1]
final_validation_score = roc_auc_score(y_va, va_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    id_col: test_df[id_col],
    target_col: test_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(submission.head())
