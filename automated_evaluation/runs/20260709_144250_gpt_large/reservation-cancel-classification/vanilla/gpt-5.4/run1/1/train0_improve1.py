
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")


def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    package_installed = True
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        package_installed = False
    if not package_installed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_package("numpy")
ensure_package("pandas")
ensure_package("sklearn", "scikit-learn")
ensure_package("xgboost")
ensure_package("catboost")
ensure_package("scipy")
ensure_package("torch")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import rankdata, spearmanr

TABPFN_IMPORTABLE = False
try:
    import importlib.util
    tabpfn_spec = importlib.util.find_spec("tabpfn")
except ImportError:
    tabpfn_spec = None

if tabpfn_spec is not None:
    try:
        ensure_package("tabpfn")
        from tabpfn import TabPFNClassifier
        import torch
        TABPFN_IMPORTABLE = True
    except Exception:
        TABPFN_IMPORTABLE = False
else:
    try:
        import torch
    except Exception:
        torch = None


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
            X[c] = X[c].fillna("missing").astype(str)
            X_test[c] = X_test[c].fillna("missing").astype(str)
        else:
            fill_value = X[c].median()
            X[c] = X[c].fillna(fill_value)
            X_test[c] = X_test[c].fillna(fill_value)

    cat_cols = [c for c in features if X[c].dtype == "object"]
    cat_idx = [features.index(c) for c in cat_cols]

    return X, y, X_test, features, cat_idx


def _rank_normalize(pred):
    pred = np.asarray(pred, dtype=float)
    if pred.size == 0:
        return pred
    ranks = rankdata(pred, method="average")
    return (ranks - 0.5) / len(ranks)


def _is_imbalanced(y, threshold=0.4):
    y_arr = np.asarray(y)
    pos_rate = y_arr.mean()
    return min(pos_rate, 1.0 - pos_rate) < threshold


def fit_xgboost_model(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test):
    model = XGBClassifier(
        booster="dart",
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.65,
        colsample_bytree=0.55,
        colsample_bylevel=0.7,
        min_child_weight=4,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        rate_drop=0.1,
        skip_drop=0.5,
        normalize_type="tree",
        sample_type="uniform",
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    valid_pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)

    model.fit(
        X_full,
        y_full,
        verbose=False
    )
    test_pred = model.predict_proba(X_test)[:, 1]
    return score, valid_pred, test_pred


def fit_catboost_model(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test, cat_idx):
    imbalanced = _is_imbalanced(y_train)

    model = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=8.0,
        random_strength=1.5,
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bernoulli",
        subsample=0.8,
        auto_class_weights="Balanced" if imbalanced else None,
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=200
    )

    valid_pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)

    best_iterations = model.get_best_iteration()
    final_iterations = best_iterations + 1 if best_iterations is not None and best_iterations > 0 else 2000

    final_model = CatBoostClassifier(
        iterations=final_iterations,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=8.0,
        random_strength=1.5,
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bernoulli",
        subsample=0.8,
        auto_class_weights="Balanced" if imbalanced else None,
        random_seed=42,
        verbose=False
    )

    final_model.fit(
        X_full,
        y_full,
        cat_features=cat_idx,
        verbose=False
    )
    test_pred = final_model.predict_proba(X_test)[:, 1]
    return score, valid_pred, test_pred


def fit_tabpfn_model(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test):
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    model = TabPFNClassifier(device=device)
    model.fit(X_train, y_train)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)

    model.fit(X_full, y_full)
    test_pred = model.predict_proba(X_test)[:, 1]
    return score, valid_pred, test_pred


def blend_model_predictions(y_valid, xgb_valid_pred, xgb_test_pred,
                            cat_valid_pred, cat_test_pred,
                            tabpfn_valid_pred=None, tabpfn_test_pred=None,
                            tabpfn_corr_threshold=0.985):
    xgb_valid_rank = _rank_normalize(xgb_valid_pred)
    cat_valid_rank = _rank_normalize(cat_valid_pred)
    xgb_test_rank = _rank_normalize(xgb_test_pred)
    cat_test_rank = _rank_normalize(cat_test_pred)

    model_preds = {
        "xgb": {
            "valid_raw": np.asarray(xgb_valid_pred),
            "test_raw": np.asarray(xgb_test_pred),
            "valid_rank": xgb_valid_rank,
            "test_rank": xgb_test_rank,
        },
        "cat": {
            "valid_raw": np.asarray(cat_valid_pred),
            "test_raw": np.asarray(cat_test_pred),
            "valid_rank": cat_valid_rank,
            "test_rank": cat_test_rank,
        },
    }

    use_tabpfn = tabpfn_valid_pred is not None and tabpfn_test_pred is not None
    if use_tabpfn:
        tabpfn_valid_rank = _rank_normalize(tabpfn_valid_pred)
        tabpfn_test_rank = _rank_normalize(tabpfn_test_pred)
        corr_with_xgb = spearmanr(xgb_valid_pred, tabpfn_valid_pred).correlation
        if np.isnan(corr_with_xgb):
            corr_with_xgb = 1.0
        if abs(corr_with_xgb) < tabpfn_corr_threshold:
            model_preds["tabpfn"] = {
                "valid_raw": np.asarray(tabpfn_valid_pred),
                "test_raw": np.asarray(tabpfn_test_pred),
                "valid_rank": tabpfn_valid_rank,
                "test_rank": tabpfn_test_rank,
            }

    selected_names = list(model_preds.keys())
    if len(selected_names) >= 3:
        pair_scores = []
        for i in range(len(selected_names)):
            for j in range(i + 1, len(selected_names)):
                n1, n2 = selected_names[i], selected_names[j]
                corr = spearmanr(model_preds[n1]["valid_raw"], model_preds[n2]["valid_raw"]).correlation
                if np.isnan(corr):
                    corr = 1.0
                pair_scores.append((abs(corr), (n1, n2)))
        pair_scores.sort(key=lambda x: x[0])
        selected_names = list(pair_scores[0][1])

    valid_blend = np.mean([model_preds[name]["valid_rank"] for name in selected_names], axis=0)
    test_blend = np.mean([model_preds[name]["test_rank"] for name in selected_names], axis=0)
    blend_score = roc_auc_score(y_valid, valid_blend)

    return blend_score, valid_blend, test_blend, selected_names


def main():
    train_path = os.path.join(".", "input", "train.csv")
    test_path = os.path.join(".", "input", "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = "booking_status"
    id_col = "id"

    X_num, y, X_test_num, _ = preprocess_for_xgb_tabpfn(train, test, target, id_col)
    X_cat, _, X_test_cat, _, cat_idx = preprocess_for_catboost(train, test, target, id_col)

    train_idx, valid_idx = train_test_split(
        np.arange(len(train)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_num = X_num.iloc[train_idx]
    X_valid_num = X_num.iloc[valid_idx]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    X_train_cat = X_cat.iloc[train_idx]
    X_valid_cat = X_cat.iloc[valid_idx]

    xgb_score, xgb_valid_pred, xgb_test_pred = fit_xgboost_model(
        X_train_num, y_train, X_valid_num, y_valid, X_num, y, X_test_num
    )

    cat_score, cat_valid_pred, cat_test_pred = fit_catboost_model(
        X_train_cat, y_train, X_valid_cat, y_valid, X_cat, y, X_test_cat, cat_idx
    )

    tab_valid_pred = None
    tab_test_pred = None
    tab_score = None

    tabpfn_enabled = TABPFN_IMPORTABLE and (os.environ.get("TABPFN_TOKEN", "").strip() != "")
    if tabpfn_enabled:
        try:
            tab_score, tab_valid_pred, tab_test_pred = fit_tabpfn_model(
                X_train_num, y_train, X_valid_num, y_valid, X_num, y, X_test_num
            )
        except Exception:
            tab_valid_pred = None
            tab_test_pred = None
            tab_score = None

    blend_score_rank, blend_valid_rank, blend_test_rank, selected_names = blend_model_predictions(
        y_valid,
        xgb_valid_pred, xgb_test_pred,
        cat_valid_pred, cat_test_pred,
        tab_valid_pred, tab_test_pred
    )

    candidate_names = ["xgb", "cat"]
    candidate_scores = [xgb_score, cat_score]
    candidate_valid_preds = [xgb_valid_pred, cat_valid_pred]
    candidate_test_preds = [xgb_test_pred, cat_test_pred]

    if tab_score is not None and tab_valid_pred is not None and tab_test_pred is not None:
        candidate_names.append("tabpfn")
        candidate_scores.append(tab_score)
        candidate_valid_preds.append(tab_valid_pred)
        candidate_test_preds.append(tab_test_pred)

    weights = np.array(candidate_scores, dtype=float)
    if np.allclose(weights.sum(), 0):
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    ensemble_valid_pred = np.zeros(len(y_valid), dtype=float)
    ensemble_test_pred = np.zeros(len(test), dtype=float)

    for w, vp, tp in zip(weights, candidate_valid_preds, candidate_test_preds):
        ensemble_valid_pred += w * vp
        ensemble_test_pred += w * tp

    weighted_score = roc_auc_score(y_valid, ensemble_valid_pred)

    if blend_score_rank >= weighted_score:
        final_validation_score = blend_score_rank
        final_test_pred = blend_test_rank
    else:
        final_validation_score = weighted_score
        final_test_pred = ensemble_test_pred

    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({
        "id": test[id_col],
        "booking_status": final_test_pred
    })
    submission.to_csv("submission_ensemble.csv", index=False)


if __name__ == "__main__":
    main()
