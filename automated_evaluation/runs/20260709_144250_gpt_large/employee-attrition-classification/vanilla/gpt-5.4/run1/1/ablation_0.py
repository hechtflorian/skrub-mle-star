
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
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
id_col = "id"

X = train.drop(columns=[target]).copy()
y = train[target].copy()
X_test = test.copy()

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

def prepare_data(X_in, fill_cats=True, fill_nums=True):
    X_out = X_in.copy()
    if fill_cats:
        for c in cat_cols:
            if c in X_out.columns:
                X_out[c] = X_out[c].fillna("Missing")
    if fill_nums:
        for c in num_cols:
            if c in X_out.columns:
                med = X_out[c].median()
                X_out[c] = X_out[c].fillna(med)
    return X_out

def encode_categoricals(train_df, valid_df=None, test_df=None):
    train_enc = train_df.copy()
    valid_enc = valid_df.copy() if valid_df is not None else None
    test_enc = test_df.copy() if test_df is not None else None

    for c in cat_cols:
        if c not in train_enc.columns:
            continue

        train_series = train_enc[c].astype(str)
        categories = pd.Index(sorted(train_series.unique().tolist()))
        mapping = {v: i for i, v in enumerate(categories)}

        train_enc[c] = train_series.map(mapping).astype(int)

        if valid_enc is not None and c in valid_enc.columns:
            valid_series = valid_enc[c].astype(str)
            valid_enc[c] = valid_series.map(mapping).fillna(-1).astype(int)

        if test_enc is not None and c in test_enc.columns:
            test_series = test_enc[c].astype(str)
            test_enc[c] = test_series.map(mapping).fillna(-1).astype(int)

    return train_enc, valid_enc, test_enc

def run_experiment(name, X_data, use_cat_features=True, use_best_model=True):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_data,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if use_cat_features:
        X_train_fit = X_train.copy()
        X_valid_fit = X_valid.copy()
        fit_cat_features = [c for c in cat_cols if c in X_train_fit.columns]
    else:
        X_train_fit, X_valid_fit, _ = encode_categoricals(X_train, X_valid, None)
        fit_cat_features = None

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0,
        random_state=42
    )

    fit_kwargs = {
        "X": X_train_fit,
        "y": y_train,
        "eval_set": (X_valid_fit, y_valid),
        "use_best_model": use_best_model
    }

    if use_cat_features:
        fit_kwargs["cat_features"] = fit_cat_features

    model.fit(**fit_kwargs)
    valid_pred = model.predict_proba(X_valid_fit)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)
    print(f"{name}: AUC={score:.6f}")
    return score

results = {}

X_base = prepare_data(X, fill_cats=True, fill_nums=True)
X_test_base = prepare_data(X_test, fill_cats=True, fill_nums=True)

results["baseline_full_pipeline"] = run_experiment(
    "baseline_full_pipeline",
    X_base,
    use_cat_features=True,
    use_best_model=True
)

results["ablation_no_best_model"] = run_experiment(
    "ablation_no_best_model",
    X_base,
    use_cat_features=True,
    use_best_model=False
)

X_no_missing = X.copy()
results["ablation_no_missing_imputation"] = run_experiment(
    "ablation_no_missing_imputation",
    X_no_missing,
    use_cat_features=True,
    use_best_model=True
)

results["ablation_no_cat_features"] = run_experiment(
    "ablation_no_cat_features",
    X_base,
    use_cat_features=False,
    use_best_model=True
)

baseline_score = results["baseline_full_pipeline"]
drops = {}
for k, v in results.items():
    if k != "baseline_full_pipeline":
        drops[k] = baseline_score - v

print("\nAblation impact relative to baseline:")
for k, v in sorted(drops.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: AUC drop = {v:.6f}")

most_important_part = max(drops, key=drops.get)
print(f"\nPart contributing most to overall performance: {most_important_part} (largest AUC drop = {drops[most_important_part]:.6f})")

X_train_final, X_valid_final, y_train_final, y_valid_final = train_test_split(
    X_base,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

final_model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
    random_state=42
)

final_model.fit(
    X_train_final,
    y_train_final,
    eval_set=(X_valid_final, y_valid_final),
    cat_features=cat_cols,
    use_best_model=True
)

final_valid_pred = final_model.predict_proba(X_valid_final)[:, 1]
final_validation_score = roc_auc_score(y_valid_final, final_valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

full_model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
    random_state=42
)

full_model.fit(
    X_base,
    y,
    cat_features=cat_cols
)

test_pred = full_model.predict_proba(X_test_base)[:, 1]
submission = pd.DataFrame({
    "EmployeeNumber": X_test_base[id_col],
    "Attrition": test_pred
})
submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
