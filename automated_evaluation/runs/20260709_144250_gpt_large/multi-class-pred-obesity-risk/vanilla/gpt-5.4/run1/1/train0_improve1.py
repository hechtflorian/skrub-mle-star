
import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        __import__(import_name)

ensure_package("catboost")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

TARGET = "NObeyesdad"
ID = "id"

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

candidate_cat_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

X = train.drop(columns=[ID, TARGET], errors="ignore").copy()
y = train[TARGET].astype(str).copy()
X_test = test.drop(columns=[ID], errors="ignore").copy()

missing_in_test = [c for c in X.columns if c not in X_test.columns]
for c in missing_in_test:
    X_test[c] = np.nan

extra_in_test = [c for c in X_test.columns if c not in X.columns]
if extra_in_test:
    X_test = X_test.drop(columns=extra_in_test)

X_test = X_test.reindex(columns=X.columns)

auto_cat_cols = [
    c for c in X.columns
    if str(X[c].dtype) == "object" or str(X[c].dtype) == "category"
]

cat_cols = sorted(set([c for c in candidate_cat_cols if c in X.columns] + auto_cat_cols))
cat_cols = [c for c in cat_cols if c in X.columns and c in X_test.columns]

for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

non_cat_cols = [c for c in X.columns if c not in cat_cols]
for col in non_cat_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
    median_value = X[col].median()
    if pd.isna(median_value):
        median_value = 0.0
    X[col] = X[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Reduced ensemble size and iterations to avoid timeout while keeping CV
seeds = [42]

class_labels = np.array(sorted(y.unique()))
label_to_index = {label: i for i, label in enumerate(class_labels)}

oof_pred_proba = np.zeros((len(X), len(class_labels)), dtype=np.float32)
test_pred_proba = np.zeros((len(X_test), len(class_labels)), dtype=np.float32)
fold_accuracies = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train = X.iloc[train_idx].copy()
    X_valid = X.iloc[valid_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_valid = y.iloc[valid_idx].copy()

    fold_valid_proba = np.zeros((len(valid_idx), len(class_labels)), dtype=np.float32)
    fold_test_proba = np.zeros((len(X_test), len(class_labels)), dtype=np.float32)

    for seed in seeds:
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="Accuracy",
            iterations=1200,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5.0,
            random_strength=1.5,
            bagging_temperature=0.5,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False
        )

        model_classes = np.array(model.classes_).astype(str)

        valid_seed_proba_raw = model.predict_proba(X_valid)
        test_seed_proba_raw = model.predict_proba(X_test)

        valid_seed_proba = np.zeros((len(X_valid), len(class_labels)), dtype=np.float32)
        test_seed_proba = np.zeros((len(X_test), len(class_labels)), dtype=np.float32)

        for i, cls in enumerate(model_classes):
            if cls in label_to_index:
                target_idx = label_to_index[cls]
                valid_seed_proba[:, target_idx] = valid_seed_proba_raw[:, i]
                test_seed_proba[:, target_idx] = test_seed_proba_raw[:, i]

        fold_valid_proba += valid_seed_proba / len(seeds)
        fold_test_proba += test_seed_proba / len(seeds)

    oof_pred_proba[valid_idx] = fold_valid_proba
    test_pred_proba += fold_test_proba / skf.n_splits

    valid_pred = class_labels[np.argmax(fold_valid_proba, axis=1)]
    fold_acc = accuracy_score(y_valid, valid_pred)
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold} Accuracy: {fold_acc:.6f}")

oof_pred = class_labels[np.argmax(oof_pred_proba, axis=1)]
final_validation_score = accuracy_score(y, oof_pred)

print(f"CV Accuracy Mean: {np.mean(fold_accuracies):.6f}")
print(f"Final OOF Validation Performance: {final_validation_score:.6f}")
print(f"Final Validation Performance: {final_validation_score}")

final_model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="Accuracy",
    iterations=1200,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5.0,
    random_strength=1.5,
    bagging_temperature=0.5,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1
)

final_model.fit(
    X,
    y,
    cat_features=cat_feature_indices,
    verbose=False
)

final_test_proba_raw = final_model.predict_proba(X_test)
final_test_proba = np.zeros((len(X_test), len(class_labels)), dtype=np.float32)
final_model_classes = np.array(final_model.classes_).astype(str)

for i, cls in enumerate(final_model_classes):
    if cls in label_to_index:
        target_idx = label_to_index[cls]
        final_test_proba[:, target_idx] = final_test_proba_raw[:, i]

cv_weight = 0.5
full_weight = 0.5
blended_test_proba = cv_weight * test_pred_proba + full_weight * final_test_proba

test_pred = class_labels[np.argmax(blended_test_proba, axis=1)]

submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred
})
submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
