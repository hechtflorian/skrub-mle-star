
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
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

valid_pred_original = model.predict_proba(X_valid)[:, 1]
original_validation_score = roc_auc_score(y_valid, valid_pred_original)

test_pred_original = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred_original
})
submission.to_csv("submission_catboost.csv", index=False)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

valid_cv_on_holdout_folds = np.zeros((len(X_valid), n_splits))
test_pred_cv_folds = np.zeros((len(X_test), n_splits))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr = X_train.iloc[tr_idx].copy()
    y_tr = y_train.iloc[tr_idx].copy()
    X_va = X_train.iloc[va_idx].copy()
    y_va = y_train.iloc[va_idx].copy()

    cv_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_state=42 + fold
    )

    cv_model.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True
    )

    valid_cv_on_holdout_folds[:, fold] = cv_model.predict_proba(X_valid)[:, 1]
    test_pred_cv_folds[:, fold] = cv_model.predict_proba(X_test)[:, 1]

valid_pred_cv_on_holdout = valid_cv_on_holdout_folds.mean(axis=1)
test_pred_cv = test_pred_cv_folds.mean(axis=1)

submission_cv = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred_cv
})
submission_cv.to_csv("submission_catboost_cv.csv", index=False)

sub1 = pd.read_csv("submission_catboost.csv")
sub2 = pd.read_csv("submission_catboost_cv.csv")

merged_test = sub1.merge(
    sub2,
    on="EmployeeNumber",
    suffixes=("_original", "_cv")
)

def rank_normalize(arr):
    s = pd.Series(arr)
    return (s.rank(method="average") / len(s)).values

def minmax01(arr):
    arr = np.asarray(arr, dtype=float)
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-15:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def rerank01(arr):
    return rank_normalize(arr)

def build_candidates(r1, r2):
    eps = 1e-9
    candidates = {}

    candidates["rank_mean_50_50"] = rerank01(0.5 * r1 + 0.5 * r2)
    candidates["rank_mean_25_75"] = rerank01(0.25 * r1 + 0.75 * r2)

    for w1, w2 in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]:
        candidates[f"rank_mean_{w1}_{w2}"] = rerank01(w1 * r1 + w2 * r2)

    for a, b in [(0.3, 0.7), (0.2, 0.8), (0.25, 0.75)]:
        geom = (r1 ** a) * (r2 ** b)
        candidates[f"geom_{a}_{b}"] = rerank01(geom)

    for w1, w2 in [(0.3, 0.7), (0.4, 0.6)]:
        harm = 1.0 / (w1 / (r1 + eps) + w2 / (r2 + eps))
        candidates[f"harm_{w1}_{w2}"] = rerank01(harm)

    avg_rank = 0.5 * r1 + 0.5 * r2
    for alpha in [0.05, 0.1, 0.15]:
        boosted = avg_rank + alpha * np.abs(r2 - r1) * np.sign(r2 - 0.5)
        candidates[f"agree_{alpha}"] = rerank01(boosted)

    return candidates

r1_valid = rank_normalize(valid_pred_original)
r2_valid = rank_normalize(valid_pred_cv_on_holdout)

valid_candidates = build_candidates(r1_valid, r2_valid)

best_name = None
best_score = -1.0
best_valid_pred = None

for name, pred in valid_candidates.items():
    score = roc_auc_score(y_valid, pred)
    if score > best_score:
        best_score = score
        best_name = name
        best_valid_pred = pred

r1_test = rank_normalize(merged_test["Attrition_original"].values)
r2_test = rank_normalize(merged_test["Attrition_cv"].values)
test_candidates = build_candidates(r1_test, r2_test)
final_test_pred = test_candidates[best_name]

final_submission = pd.DataFrame({
    "EmployeeNumber": merged_test["EmployeeNumber"],
    "Attrition": final_test_pred
})
final_submission.to_csv("submission.csv", index=False)

final_validation_score = best_score
print(f"Best Blend: {best_name}")
print(f"Original Validation Performance: {original_validation_score:.6f}")
print(f"Final Validation Performance: {final_validation_score}")
