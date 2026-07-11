
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "booking_status"
id_col = "id"

X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].astype(int).copy()
X_test = test_df.copy()

# Remove id from features
X = X.drop(columns=[id_col])
X_test = X_test.drop(columns=[id_col])

# Identify categorical columns:
# 1) object/category columns
# 2) low-cardinality integer columns
cat_cols = []
for col in X.columns:
    if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
        cat_cols.append(col)
    elif X[col].nunique(dropna=False) <= 20:
        cat_cols.append(col)

cat_cols = list(dict.fromkeys(cat_cols))  # deduplicate while preserving order

# Cast categorical columns to category dtype for LightGBM
for c in cat_cols:
    if c in X.columns:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

# Use fewer folds and lighter models to avoid timeout
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/{n_splits}")

    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,          # keep subsampling
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42 + fold,
        n_jobs=-1,
        class_weight="balanced"
    )

    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )

    va_pred = lgb_model.predict_proba(X_va)[:, 1]
    oof_lgb[va_idx] = va_pred

    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits

oof_auc = roc_auc_score(y, oof_lgb)
print(f"OOF LightGBM AUC: {oof_auc}")
final_validation_score = oof_auc
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    id_col: test_df[id_col],
    target_col: test_lgb
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(submission.head())
