
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")


def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_package("numpy")
ensure_package("pandas")
ensure_package("sklearn", "scikit-learn")
ensure_package("xgboost")
ensure_package("torch")
ensure_package("tabpfn")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

TABPFN_AVAILABLE = True
try:
    from tabpfn import TabPFNClassifier
except Exception:
    TABPFN_AVAILABLE = False


def preprocess_data(train_df, test_df, target_col, id_col):
    features = [c for c in train_df.columns if c not in [id_col, target_col]]

    X = train_df[features].copy()
    y = train_df[target_col].copy()
    X_test = test_df[features].copy()

    object_cols = [c for c in features if X[c].dtype == "object" or X_test[c].dtype == "object"]

    for c in object_cols:
        le = LabelEncoder()
        combined = pd.concat([X[c].astype(str), X_test[c].astype(str)], axis=0)
        le.fit(combined)
        X[c] = le.transform(X[c].astype(str))
        X_test[c] = le.transform(X_test[c].astype(str))

    for c in features:
        if X[c].isnull().any() or X_test[c].isnull().any():
            if pd.api.types.is_numeric_dtype(X[c]):
                fill_value = X[c].median()
            else:
                fill_value = "missing"
            X[c] = X[c].fillna(fill_value)
            X_test[c] = X_test[c].fillna(fill_value)

    X = X.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    for c in features:
        if X[c].isnull().any() or X_test[c].isnull().any():
            fill_value = X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else 0
            X[c] = X[c].fillna(fill_value)
            X_test[c] = X_test[c].fillna(fill_value)

    return X, y, X_test, features


def fit_tabpfn_if_possible(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test):
    if not TABPFN_AVAILABLE:
        raise RuntimeError("TabPFN is not available.")

    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TABPFN_TOKEN is not set.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNClassifier(device=device)
    model.fit(X_train, y_train)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)

    model.fit(X_full, y_full)
    test_pred = model.predict_proba(X_test)[:, 1]
    return score, test_pred


def fit_xgboost_fallback(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test):
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)
    valid_pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)

    model.fit(X_full, y_full)
    test_pred = model.predict_proba(X_test)[:, 1]
    return score, test_pred


def main():
    train_path = os.path.join(".", "input", "train.csv")
    test_path = os.path.join(".", "input", "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = "booking_status"
    id_col = "id"

    X, y, X_test, features = preprocess_data(train, test, target, id_col)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    final_validation_score = None
    test_pred = None

    tabpfn_attempted = False
    if TABPFN_AVAILABLE:
        try:
            tabpfn_attempted = True
            final_validation_score, test_pred = fit_tabpfn_if_possible(
                X_train, y_train, X_valid, y_valid, X, y, X_test
            )
        except Exception as e:
            print(f"TabPFN unavailable or failed, falling back to XGBoost. Reason: {str(e)}")

    if final_validation_score is None or test_pred is None:
        final_validation_score, test_pred = fit_xgboost_fallback(
            X_train, y_train, X_valid, y_valid, X, y, X_test
        )

    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({
        "id": test[id_col],
        "booking_status": test_pred
    })
    submission.to_csv("submission_tabpfn.csv", index=False)
    print("Saved submission to submission_tabpfn.csv")


if __name__ == "__main__":
    main()
