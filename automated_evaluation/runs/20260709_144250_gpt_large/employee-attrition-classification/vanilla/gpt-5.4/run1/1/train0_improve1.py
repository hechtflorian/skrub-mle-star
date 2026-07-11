
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
num_cols = [c for c in X.columns if c not in cat_cols]

X = X.copy()
X_test = X_test.copy()

for c in cat_cols:
    X[c] = X[c].fillna("Missing").astype(str)
    X_test[c] = X_test[c].fillna("Missing").astype(str)

for c in num_cols:
    med = X[c].median()
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

from sklearn.model_selection import StratifiedKFold
import numpy as np

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
fold_scores = []
models = []

rare_min_count = max(10, int(0.005 * len(X)))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train = X.iloc[train_idx].copy()
    X_valid = X.iloc[valid_idx].copy()
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    for c in cat_cols:
        value_counts = X_train[c].value_counts(dropna=False)
        rare_categories = value_counts[value_counts < rare_min_count].index

        X_train[c] = X_train[c].replace(rare_categories, "Rare")
        X_valid[c] = X_valid[c].where(~X_valid[c].isin(rare_categories), "Rare")
        X_test_fold_col = X_test[c].where(~X_test[c].isin(rare_categories), "Rare")

        known_categories = set(X_train[c].unique())
        X_valid[c] = X_valid[c].where(X_valid[c].isin(known_categories), "Rare")

        X_train[c] = X_train[c].astype(str)
        X_valid[c] = X_valid[c].astype(str)

        if fold == 1:
            X_test[c] = X_test_fold_col.where(X_test_fold_col.isin(known_categories), "Rare").astype(str)
        else:
            X_test_fold_col = X_test_fold_col.where(X_test_fold_col.isin(known_categories), "Rare").astype(str)

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=200,
        random_state=42 + fold,
        auto_class_weights="Balanced",
        bagging_temperature=1.0,
        random_strength=1.0,
        od_type="Iter",
        od_wait=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    valid_pred = model.predict_proba(X_valid)[:, 1]
    oof_pred[valid_idx] = valid_pred
    fold_scores.append(roc_auc_score(y_valid, valid_pred))

    X_test_infer = X_test.copy()
    for c in cat_cols:
        value_counts = X_train[c].value_counts(dropna=False)
        rare_categories = value_counts[value_counts < rare_min_count].index
        X_test_infer[c] = X_test_infer[c].replace(rare_categories, "Rare")
        known_categories = set(X_train[c].unique())
        X_test_infer[c] = X_test_infer[c].where(X_test_infer[c].isin(known_categories), "Rare").astype(str)

    test_pred += model.predict_proba(X_test_infer)[:, 1] / skf.n_splits
    models.append(model)

final_validation_score = roc_auc_score(y, oof_pred)


submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred
})
submission.to_csv("submission_catboost.csv", index=False)

print(f"Final Validation Performance: {final_validation_score:.6f}")
