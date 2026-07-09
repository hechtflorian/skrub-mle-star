
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SUBMISSION_PATH = "./final/submission.csv"

os.makedirs("./final", exist_ok=True)

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

# Identify likely categorical columns:
# 1) object columns
# 2) low-cardinality integer-like columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for col in X.columns:
    if col not in cat_cols and X[col].nunique(dropna=False) <= 20:
        cat_cols.append(col)

# Ensure unique order
cat_cols = list(dict.fromkeys(cat_cols))

# 5-fold OOF setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

lgb_oof_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    X_te_lgb = X_test.copy()

    for c in cat_cols:
        if c in X_tr_lgb.columns:
            X_tr_lgb[c] = X_tr_lgb[c].astype("category")
            X_va_lgb[c] = X_va_lgb[c].astype("category")
            X_te_lgb[c] = X_te_lgb[c].astype("category")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )

    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0)
        ]
    )

    lgb_va_pred = lgb_model.predict_proba(X_va_lgb)[:, 1]
    lgb_test_pred = lgb_model.predict_proba(X_te_lgb)[:, 1]

    oof_lgb[va_idx] = lgb_va_pred
    test_lgb += lgb_test_pred / skf.n_splits

    fold_auc = roc_auc_score(y_va, lgb_va_pred)
    lgb_oof_scores.append(fold_auc)

# OOF AUC
auc_lgb = roc_auc_score(y, oof_lgb)

# Rank transform for robustness
r_lgb = rankdata(oof_lgb, method="average") / len(oof_lgb)
r_test_lgb = rankdata(test_lgb, method="average") / len(test_lgb)

# Blend raw and rank versions lightly for stability
oof_blend = 0.7 * oof_lgb + 0.3 * r_lgb
test_blend = 0.7 * test_lgb + 0.3 * r_test_lgb

final_validation_score = roc_auc_score(y, oof_blend)
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    id_col: test_df[id_col],
    target_col: test_blend
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(submission.head())
