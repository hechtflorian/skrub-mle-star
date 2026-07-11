
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
ensure_package("torch")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TABPFN_IMPORTABLE = False
tabpfn_spec = None
try:
    import importlib.util
    tabpfn_spec = importlib.util.find_spec("tabpfn")
except ImportError:
    tabpfn_spec = None
if tabpfn_spec is not None:
    ensure_package("tabpfn")
    from tabpfn import TabPFNClassifier
    import torch
    TABPFN_IMPORTABLE = True


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


def fit_xgboost_model(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test):
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
    valid_pred = np.zeros(len(X_valid), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    for seed in seeds:
        model = XGBClassifier(
            **xgb_params,
            random_state=seed
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            valid_pred += model.predict_proba(X_valid, iteration_range=(0, best_iteration + 1))[:, 1] / len(seeds)
        else:
            valid_pred += model.predict_proba(X_valid)[:, 1] / len(seeds)

        best_n_estimators = (best_iteration + 1) if best_iteration is not None else xgb_params["n_estimators"]

        full_model = XGBClassifier(
            **{**xgb_params, "n_estimators": best_n_estimators},
            random_state=seed
        )
        full_model.fit(X_full, y_full, verbose=False)
        test_pred += full_model.predict_proba(X_test)[:, 1] / len(seeds)

    score = roc_auc_score(y_valid, valid_pred)
    return score, valid_pred, test_pred


def fit_catboost_model(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test, cat_idx):
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

    best_iteration = model.get_best_iteration()
    full_iterations = best_iteration + 1 if best_iteration is not None and best_iteration >= 0 else 2000

    full_model = CatBoostClassifier(
        iterations=full_iterations,
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

    full_model.fit(
        X_full,
        y_full,
        cat_features=cat_idx,
        verbose=False
    )
    test_pred = full_model.predict_proba(X_test)[:, 1]
    return score, valid_pred, test_pred


def fit_tabpfn_model(X_train, y_train, X_valid, y_valid, X_full, y_full, X_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNClassifier(device=device)
    model.fit(X_train, y_train)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)

    model.fit(X_full, y_full)
    test_pred = model.predict_proba(X_test)[:, 1]
    return score, valid_pred, test_pred


def rank_normalize(pred):
    s = pd.Series(pred)
    ranks = s.rank(method="average").to_numpy(dtype=float)
    if len(ranks) <= 1:
        return np.zeros_like(ranks, dtype=float)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def spearman_corr(a, b):
    ra = rank_normalize(a)
    rb = rank_normalize(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def blend_model_predictions(model_names, model_scores, valid_preds, test_preds, y_valid):
    candidates = []

    weights = np.array(model_scores, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    weighted_raw_valid = np.zeros(len(y_valid), dtype=float)
    weighted_raw_test = np.zeros(len(test_preds[0]), dtype=float)
    for w, vp, tp in zip(weights, valid_preds, test_preds):
        weighted_raw_valid += w * vp
        weighted_raw_test += w * tp
    candidates.append(("weighted_raw_blend", weighted_raw_valid, weighted_raw_test))

    rank_valid_preds = [rank_normalize(vp) for vp in valid_preds]
    rank_test_preds = [rank_normalize(tp) for tp in test_preds]

    equal_rank_valid = np.mean(rank_valid_preds, axis=0)
    equal_rank_test = np.mean(rank_test_preds, axis=0)
    candidates.append(("equal_rank_blend", equal_rank_valid, equal_rank_test))

    if len(model_names) == 2:
        diverse_rank_valid = np.mean([rank_valid_preds[0], rank_valid_preds[1]], axis=0)
        diverse_rank_test = np.mean([rank_test_preds[0], rank_test_preds[1]], axis=0)
        candidates.append(("diverse_rank_blend", diverse_rank_valid, diverse_rank_test))
    else:
        corr_pairs = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                corr_ij = spearman_corr(valid_preds[i], valid_preds[j])
                corr_pairs.append((corr_ij, i, j))

        diverse_indices = None

        name_to_idx = {name: idx for idx, name in enumerate(model_names)}
        if "tabpfn" in name_to_idx:
            tab_idx = name_to_idx["tabpfn"]
            tab_corrs = []
            for i in range(len(model_names)):
                if i != tab_idx:
                    tab_corrs.append((spearman_corr(valid_preds[tab_idx], valid_preds[i]), i))
            if len(tab_corrs) > 0 and max(c for c, _ in tab_corrs) >= 0.995:
                non_tab_indices = [i for i in range(len(model_names)) if i != tab_idx]
                diverse_indices = non_tab_indices

        if diverse_indices is None:
            least_corr_pair = sorted(corr_pairs, key=lambda x: x[0])[0]
            diverse_indices = [least_corr_pair[1], least_corr_pair[2]]

        diverse_rank_valid = np.mean([rank_valid_preds[i] for i in diverse_indices], axis=0)
        diverse_rank_test = np.mean([rank_test_preds[i] for i in diverse_indices], axis=0)
        candidates.append(("diverse_rank_blend", diverse_rank_valid, diverse_rank_test))

    best_name = None
    best_valid_pred = None
    best_test_pred = None
    best_score = -1.0

    for name, vp, tp in candidates:
        score = roc_auc_score(y_valid, vp)
        if score > best_score:
            best_score = score
            best_name = name
            best_valid_pred = vp
            best_test_pred = tp

    return best_name, best_score, best_valid_pred, best_test_pred


def main():
    train_path = os.path.join(".", "input", "train.csv")
    test_path = os.path.join(".", "input", "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = "booking_status"
    id_col = "id"

    X_num, y, X_test_num, _ = preprocess_for_xgb_tabpfn(train, test, target, id_col)
    X_cat, _, X_test_cat, features_cat, cat_idx = preprocess_for_catboost(train, test, target, id_col)

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

    model_names = []
    model_scores = []
    valid_preds = []
    test_preds = []

    xgb_score, xgb_valid_pred, xgb_test_pred = fit_xgboost_model(
        X_train_num, y_train, X_valid_num, y_valid, X_num, y, X_test_num
    )
    model_names.append("xgb")
    model_scores.append(xgb_score)
    valid_preds.append(xgb_valid_pred)
    test_preds.append(xgb_test_pred)

    cat_score, cat_valid_pred, cat_test_pred = fit_catboost_model(
        X_train_cat, y_train, X_valid_cat, y_valid, X_cat, y, X_test_cat, cat_idx
    )
    model_names.append("cat")
    model_scores.append(cat_score)
    valid_preds.append(cat_valid_pred)
    test_preds.append(cat_test_pred)

    tabpfn_enabled = TABPFN_IMPORTABLE and (os.environ.get("TABPFN_TOKEN", "").strip() != "")
    if tabpfn_enabled:
        tab_score, tab_valid_pred, tab_test_pred = fit_tabpfn_model(
            X_train_num, y_train, X_valid_num, y_valid, X_num, y, X_test_num
        )
        model_names.append("tabpfn")
        model_scores.append(tab_score)
        valid_preds.append(tab_valid_pred)
        test_preds.append(tab_test_pred)

    best_blend_name, final_validation_score, ensemble_valid_pred, ensemble_test_pred = blend_model_predictions(
        model_names=model_names,
        model_scores=model_scores,
        valid_preds=valid_preds,
        test_preds=test_preds,
        y_valid=y_valid
    )

    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({
        "id": test[id_col],
        "booking_status": ensemble_test_pred
    })
    submission.to_csv("submission_ensemble.csv", index=False)


if __name__ == "__main__":
    main()
