
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

# -----------------------------
# Original holdout CatBoost script
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_original = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    random_state=42
)

model_original.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

valid_pred_original = model_original.predict_proba(X_valid)[:, 1]
original_validation_score = roc_auc_score(y_valid, valid_pred_original)

test_pred_original = model_original.predict_proba(X_test)[:, 1]

submission_original = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred_original
})
submission_original.to_csv("submission_catboost.csv", index=False)

valid_original_df = pd.DataFrame({
    "row_index": X_valid.index,
    "y_true": y_valid.values,
    "pred_original": valid_pred_original
})
valid_original_df.to_csv("validation_catboost.csv", index=False)

# -----------------------------
# Cross-validated CatBoost variant
# -----------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_pred_cv = np.zeros(len(X))
test_pred_cv = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[train_idx].copy()
    y_tr = y.iloc[train_idx].copy()
    X_va = X.iloc[valid_idx].copy()
    y_va = y.iloc[valid_idx].copy()

    model_cv = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=200,
        random_state=42 + fold
    )

    model_cv.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True
    )

    oof_pred_cv[valid_idx] = model_cv.predict_proba(X_va)[:, 1]
    test_pred_cv += model_cv.predict_proba(X_test)[:, 1] / n_splits

cv_validation_score = roc_auc_score(y, oof_pred_cv)

submission_cv = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred_cv
})
submission_cv.to_csv("submission_catboost_cv.csv", index=False)

oof_cv_df = pd.DataFrame({
    "row_index": X.index,
    "y_true": y.values,
    "pred_cv_oof": oof_pred_cv
})
oof_cv_df.to_csv("oof_catboost_cv.csv", index=False)

# -----------------------------
# Validation-side blending on common holdout view
# -----------------------------
valid_cv_on_holdout = oof_pred_cv[X_valid.index]
y_valid_aligned = y.loc[X_valid.index].values

def rank_normalize(arr):
    return pd.Series(arr).rank(method="average").values / len(arr)

candidate_scores = []

weight_grid = [(0.5, 0.5), (0.4, 0.6), (0.3, 0.7)]
for w_orig, w_cv in weight_grid:
    blended_valid = w_orig * valid_pred_original + w_cv * valid_cv_on_holdout
    score = roc_auc_score(y_valid_aligned, blended_valid)
    candidate_scores.append(("raw", w_orig, w_cv, score))

valid_rank_original = rank_normalize(valid_pred_original)
valid_rank_cv = rank_normalize(valid_cv_on_holdout)

for w_orig, w_cv in weight_grid:
    blended_valid_rank = w_orig * valid_rank_original + w_cv * valid_rank_cv
    score = roc_auc_score(y_valid_aligned, blended_valid_rank)
    candidate_scores.append(("rank", w_orig, w_cv, score))

best_mode, best_w_orig, best_w_cv, final_validation_score = max(candidate_scores, key=lambda x: x[3])

# -----------------------------
# Tiny merge script behavior
# -----------------------------
sub1 = pd.read_csv("submission_catboost.csv")
sub2 = pd.read_csv("submission_catboost_cv.csv")

merged = sub1.merge(
    sub2,
    on="EmployeeNumber",
    suffixes=("_original", "_cv")
)

if best_mode == "raw":
    merged["Attrition"] = (
        best_w_orig * merged["Attrition_original"] +
        best_w_cv * merged["Attrition_cv"]
    )
else:
    rank_orig_test = merged["Attrition_original"].rank(method="average") / len(merged)
    rank_cv_test = merged["Attrition_cv"].rank(method="average") / len(merged)
    merged["Attrition"] = (
        best_w_orig * rank_orig_test +
        best_w_cv * rank_cv_test
    )

submission_final = merged[["EmployeeNumber", "Attrition"]].copy()
submission_final.to_csv("submission.csv", index=False)

print(f"Original Holdout AUC: {original_validation_score:.6f}")
print(f"CV OOF AUC: {cv_validation_score:.6f}")
print(f"Best Blend Mode: {best_mode}, Weights: original={best_w_orig}, cv={best_w_cv}")
print(f"Final Validation Performance: {final_validation_score:.6f}")
