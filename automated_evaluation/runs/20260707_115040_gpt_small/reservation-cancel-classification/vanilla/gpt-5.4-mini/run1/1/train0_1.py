
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
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

# Identify categorical columns
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

# Prepare copies for LightGBM and CatBoost
X_tr_lgb = X_tr.copy()
X_va_lgb = X_va.copy()
X_test_lgb = X_test.copy()

for c in cat_cols:
    if c in X_tr_lgb.columns:
        X_tr_lgb[c] = X_tr_lgb[c].astype("category")
        X_va_lgb[c] = X_va_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

# LightGBM model
lgb_model = lgb.LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_tr_lgb,
    y_tr,
    eval_set=[(X_va_lgb, y_va)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]
)

# CatBoost model
cat_model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    random_seed=42,
    auto_class_weights="Balanced"
)

cat_model.fit(
    X_tr,
    y_tr,
    cat_features=[c for c in cat_cols if c in X_tr.columns],
    eval_set=(X_va, y_va),
    use_best_model=True
)

# Validation predictions
lgb_va_pred = lgb_model.predict_proba(X_va_lgb)[:, 1]
cat_va_pred = cat_model.predict_proba(X_va)[:, 1]
va_pred = 0.5 * lgb_va_pred + 0.5 * cat_va_pred

final_validation_score = roc_auc_score(y_va, va_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Test predictions
lgb_test_pred = lgb_model.predict_proba(X_test_lgb)[:, 1]
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]
test_pred = 0.5 * lgb_test_pred + 0.5 * cat_test_pred

submission = pd.DataFrame({
    id_col: test_df[id_col],
    target_col: test_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(submission.head())
