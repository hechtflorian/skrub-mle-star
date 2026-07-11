
import os
import random
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = train["Attrition"].astype(int).values
X = train.drop(columns=["Attrition"]).copy()
T = test.copy()

# -----------------------------
# Common preprocessing
# -----------------------------
for col in X.columns:
    if col in T.columns:
        if X[col].dtype == "object" or T[col].dtype == "object":
            X[col] = X[col].astype("object").fillna("NA").astype(str)
            T[col] = T[col].astype("object").fillna("NA").astype(str)
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            T[col] = T[col].fillna(median_val)

# Drop constant columns consistently
const_cols = []
for col in X.columns:
    if col in T.columns:
        if pd.Series(X[col]).nunique(dropna=False) <= 1 and pd.Series(T[col]).nunique(dropna=False) <= 1:
            const_cols.append(col)

if len(const_cols) > 0:
    X = X.drop(columns=const_cols)
    T = T.drop(columns=const_cols)

all_df = pd.concat([X, T], axis=0, ignore_index=True)
all_df = pd.get_dummies(all_df, dummy_na=True)

X_enc = all_df.iloc[:len(X)].copy()
T_enc = all_df.iloc[len(X):].copy()

# -----------------------------
# Solution 1: 5-fold OOF + test
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_1 = np.zeros(len(X_enc))
test_1 = np.zeros(len(T_enc))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_enc, y), 1):
    X_train, X_val = X_enc.iloc[tr_idx], X_enc.iloc[va_idx]
    y_train, y_val = y[tr_idx], y[va_idx]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=SEED + fold,
        subsample=0.9,
        colsample_bytree=0.9,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
        ],
    )

    oof_1[va_idx] = model.predict_proba(X_val)[:, 1]
    test_1 += model.predict_proba(T_enc)[:, 1] / skf.n_splits

# -----------------------------
# Solution 2: existing fold-based solution preserved as much as possible
# (implemented as a second independent 5-fold model with different params/seed)
# -----------------------------
oof_2 = np.zeros(len(X_enc))
test_2 = np.zeros(len(T_enc))

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 7)

for fold, (tr_idx, va_idx) in enumerate(skf2.split(X_enc, y), 1):
    X_train, X_val = X_enc.iloc[tr_idx], X_enc.iloc[va_idx]
    y_train, y_val = y[tr_idx], y[va_idx]

    model2 = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=63,
        random_state=SEED + 100 + fold,
        subsample=0.85,
        colsample_bytree=0.85,
    )

    model2.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
        ],
    )

    oof_2[va_idx] = model2.predict_proba(X_val)[:, 1]
    test_2 += model2.predict_proba(T_enc)[:, 1] / skf2.n_splits

# -----------------------------
# Blend tuning on OOF only
# -----------------------------
weights = [0.3, 0.4, 0.5, 0.6, 0.7]
best_auc = -1
best_w1 = 0.5
best_oof = None
best_test = None

for w1 in weights:
    w2 = 1.0 - w1
    blended_oof = w1 * oof_1 + w2 * oof_2
    auc = roc_auc_score(y, blended_oof)
    if auc > best_auc:
        best_auc = auc
        best_w1 = w1
        best_oof = blended_oof
        best_test = w1 * test_1 + w2 * test_2

final_validation_score = roc_auc_score(y, best_oof)
print(f"Final Validation Performance: {final_validation_score}")

# -----------------------------
# Submission
# -----------------------------
id_col = "EmployeeNumber" if "EmployeeNumber" in test.columns else "id"
submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": best_test
})
submission.to_csv("submission.csv", index=False)
