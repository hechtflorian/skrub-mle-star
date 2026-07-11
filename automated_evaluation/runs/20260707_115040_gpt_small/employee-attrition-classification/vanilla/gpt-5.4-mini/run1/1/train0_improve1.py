
import os
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
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

# Keep identifiers for submission
test_ids = T["id"].copy()

# Basic preprocessing: combine train/test for consistent categorical handling
all_data = pd.concat([X, T], axis=0, ignore_index=True)

# Drop obvious non-informative constant columns if present
for col in ["EmployeeCount", "StandardHours", "Over18"]:
    if col in all_data.columns:
        all_data.drop(columns=[col], inplace=True)

# Identify categorical columns
cat_cols = all_data.select_dtypes(include=["object"]).columns.tolist()

# Fill missing values
for col in all_data.columns:
    if col in cat_cols:
        all_data[col] = all_data[col].astype("string").fillna("missing")
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Convert categoricals to category codes using combined data
for col in cat_cols:
    all_data[col] = all_data[col].astype("category")
    all_data[col] = all_data[col].cat.codes

# Split back
X_proc = all_data.iloc[: len(X)].copy()
T_proc = all_data.iloc[len(X) :].copy()

# LightGBM model with cross-validation
params = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_pred = np.zeros(len(X_proc))
test_pred = np.zeros(len(T_proc))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_proc, y), 1):
    X_tr, X_va = X_proc.iloc[tr_idx], X_proc.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[],
    )

    oof_pred[va_idx] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(T_proc)[:, 1] / skf.n_splits

final_validation_score = roc_auc_score(y, oof_pred)
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    "EmployeeNumber": test_ids,
    "Attrition": test_pred
})

submission.to_csv("submission.csv", index=False)
