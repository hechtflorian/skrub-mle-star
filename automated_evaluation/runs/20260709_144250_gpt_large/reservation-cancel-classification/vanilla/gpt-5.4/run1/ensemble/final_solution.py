
import os
import sys
import subprocess
import warnings
import importlib.util

warnings.filterwarnings("ignore")


def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_package("numpy")
ensure_package("pandas")
ensure_package("sklearn", "scikit-learn")
ensure_package("xgboost")
ensure_package("catboost")
ensure_package("torch")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

tabpfn_spec = importlib.util.find_spec("tabpfn")
TABPFN_IMPORTABLE = tabpfn_spec is not None
if TABPFN_IMPORTABLE:
    ensure_package("tabpfn")
    from tabpfn import TabPFNClassifier
    import torch


def preprocess_for_xgb_tabpfn(train_df, test_df, target_col, id_col):
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
        needs_fill = X[c].isnull().any() or X_test[c].isnull().any()
        if needs_fill:
            fill_value = X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else "missing"
            X[c] = X[c].fillna(fill_value)
            X_test[c] = X_test[c].fillna(fill_value)

    X = X.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    for c in features:
        needs_fill = X[c].isnull().any() or X_test[c].isnull().any()
        if needs_fill:
            fill_value = X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else 0
            X[c] = X[c].fillna(fill_value)
            X_test[c] = X_test[c].fillna(fill_value)

    return X, y, X_test, features


def preprocess_for_catboost(train_df, test_df, target_col, id_col):
    features = [c for c in train_df.columns if c not in [id_col, target_col]]

    X = train_df[features].copy()
    y = train_df[target_col].copy()
    X_test = test_df[features].copy()

    for c in features:
        if X[c].dtype == "object" or X_test[c].dtype == "object":
            X[c] = X[c].astype(str).fillna("missing")
            X_test[c] = X_test[c].astype(str).fillna("missing")
        else:
            fill_value = X[c].median()
            X[c] = X[c].fillna(fill_value)
            X_test[c] = X_test[c].fillna(fill_value)

    cat_cols = [c for c in features if X[c].dtype == "object"]
    cat_idx = [features.index(c) for c in cat_cols]

    return X, y, X_test, features, cat_idx


def fit_xgboost_full(X_full, y_full, X_test):
    xgb_params = dict(
        n_estimators=3000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        tree_method="hist"
    )

    seeds = [42, 52, 62]
    test_pred = np.zeros(len(X_test), dtype=float)

    for seed in seeds:
        model = XGBClassifier(**xgb_params, random_state=seed)
        model.fit(X_full, y_full, verbose=False)
        test_pred += model.predict_proba(X_test)[:, 1] / len(seeds)

    return test_pred


def fit_catboost_full(X_full, y_full, X_test, cat_idx):
    model = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )

    model.fit(X_full, y_full, cat_features=cat_idx, verbose=False)
    test_pred = model.predict_proba(X_test)[:, 1]
    return test_pred


def fit_tabpfn_full(X_full, y_full, X_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNClassifier(device=device)
    model.fit(X_full, y_full)
    test_pred = model.predict_proba(X_test)[:, 1]
    return test_pred


def main():
    train_path = os.path.join(".", "input", "train.csv")
    test_path = os.path.join(".", "input", "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = "booking_status"
    id_col = "id"

    X_num, y, X_test_num, _ = preprocess_for_xgb_tabpfn(train, test, target, id_col)
    X_cat, _, X_test_cat, _, cat_idx = preprocess_for_catboost(train, test, target, id_col)

    preds = []

    xgb_test_pred = fit_xgboost_full(X_num, y, X_test_num)
    preds.append(xgb_test_pred)

    cat_test_pred = fit_catboost_full(X_cat, y, X_test_cat, cat_idx)
    preds.append(cat_test_pred)

    tabpfn_token = os.environ.get("TABPFN_TOKEN", "").strip()
    if TABPFN_IMPORTABLE and len(tabpfn_token) > 0:
        tab_test_pred = fit_tabpfn_full(X_num, y, X_test_num)
        preds.append(tab_test_pred)

    final_pred = np.mean(np.column_stack(preds), axis=1)

    os.makedirs("./final", exist_ok=True)

    submission = pd.DataFrame({
        "id": test[id_col],
        "booking_status": final_pred
    })
    submission.to_csv("./final/submission.csv", index=False)


if __name__ == "__main__":
    main()
