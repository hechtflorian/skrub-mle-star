import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

try:
    import importlib.util
    TABPFN_AVAILABLE = importlib.util.find_spec("tabpfn") is not None
    if TABPFN_AVAILABLE:
        from tabpfn import TabPFNClassifier
        import torch
except Exception:
    TABPFN_AVAILABLE = False


def preprocess_for_xgb(train_df, target_col, id_col, use_label_encoding=True, use_missing_fill=True):
    features = [c for c in train_df.columns if c not in [id_col, target_col]]
    X = train_df[features].copy()
    y = train_df[target_col].copy()

    object_cols = [c for c in features if X[c].dtype == "object"]

    if use_label_encoding:
        for c in object_cols:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
    else:
        for c in object_cols:
            X[c] = X[c].astype("category").cat.codes

    X = X.apply(pd.to_numeric, errors="coerce")

    if use_missing_fill:
        for c in features:
            if X[c].isnull().any():
                fill_value = X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else 0
                X[c] = X[c].fillna(fill_value)

    return X, y


def preprocess_for_catboost(train_df, target_col, id_col, use_categorical_native=True, use_missing_fill=True):
    features = [c for c in train_df.columns if c not in [id_col, target_col]]
    X = train_df[features].copy()
    y = train_df[target_col].copy()

    if use_categorical_native:
        for c in features:
            if X[c].dtype == "object":
                X[c] = X[c].astype(str).fillna("missing")
            elif use_missing_fill:
                X[c] = X[c].fillna(X[c].median())
        cat_cols = [c for c in features if X[c].dtype == "object"]
        cat_idx = [features.index(c) for c in cat_cols]
    else:
        for c in features:
            if X[c].dtype == "object":
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))
            if use_missing_fill and X[c].isnull().any():
                X[c] = X[c].fillna(X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else 0)
        X = X.apply(pd.to_numeric, errors="coerce")
        if use_missing_fill:
            for c in features:
                if X[c].isnull().any():
                    X[c] = X[c].fillna(X[c].median())
        cat_idx = []

    return X, y, cat_idx


def fit_xgb(X_train, y_train, X_valid, y_valid):
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
    pred = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, pred), pred


def fit_catboost(X_train, y_train, X_valid, y_valid, cat_idx):
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=False
    )
    pred = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, pred), pred


def fit_tabpfn(X_train, y_train, X_valid, y_valid):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNClassifier(device=device)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, pred), pred


def evaluate_pipeline(train_df, target_col, id_col, config):
    y = train_df[target_col].copy()
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scores = {}
    valid_preds = []
    model_scores = []

    if config["use_xgb"]:
        X_xgb, y_xgb = preprocess_for_xgb(
            train_df,
            target_col,
            id_col,
            use_label_encoding=config["xgb_label_encoding"],
            use_missing_fill=config["xgb_missing_fill"]
        )
        xgb_score, xgb_pred = fit_xgb(
            X_xgb.iloc[train_idx], y_xgb.iloc[train_idx],
            X_xgb.iloc[valid_idx], y_xgb.iloc[valid_idx]
        )
        scores["xgb"] = xgb_score
        valid_preds.append(xgb_pred)
        model_scores.append(xgb_score)

    if config["use_cat"]:
        X_cat, y_cat, cat_idx = preprocess_for_catboost(
            train_df,
            target_col,
            id_col,
            use_categorical_native=config["cat_native_categorical"],
            use_missing_fill=config["cat_missing_fill"]
        )
        cat_score, cat_pred = fit_catboost(
            X_cat.iloc[train_idx], y_cat.iloc[train_idx],
            X_cat.iloc[valid_idx], y_cat.iloc[valid_idx],
            cat_idx
        )
        scores["cat"] = cat_score
        valid_preds.append(cat_pred)
        model_scores.append(cat_score)

    if config["use_tabpfn"] and TABPFN_AVAILABLE:
        X_tab, y_tab = preprocess_for_xgb(
            train_df,
            target_col,
            id_col,
            use_label_encoding=True,
            use_missing_fill=True
        )
        tab_score, tab_pred = fit_tabpfn(
            X_tab.iloc[train_idx], y_tab.iloc[train_idx],
            X_tab.iloc[valid_idx], y_tab.iloc[valid_idx]
        )
        scores["tabpfn"] = tab_score
        valid_preds.append(tab_pred)
        model_scores.append(tab_score)

    if len(valid_preds) == 0:
        return None, scores

    if config["weighted_ensemble"]:
        weights = np.array(model_scores, dtype=float)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(valid_preds), dtype=float) / len(valid_preds)

    ensemble_pred = np.zeros(len(valid_idx), dtype=float)
    for w, pred in zip(weights, valid_preds):
        ensemble_pred += w * pred

    ensemble_score = roc_auc_score(y.iloc[valid_idx], ensemble_pred)
    return ensemble_score, scores


def main():
    train_path = os.path.join(".", "input", "train.csv")
    train = pd.read_csv(train_path)

    target_col = "booking_status"
    id_col = "id"

    use_tabpfn = TABPFN_AVAILABLE and (os.environ.get("TABPFN_TOKEN", "").strip() != "")

    ablations = {
        "baseline_full_pipeline": {
            "use_xgb": True,
            "use_cat": True,
            "use_tabpfn": use_tabpfn,
            "xgb_label_encoding": True,
            "xgb_missing_fill": True,
            "cat_native_categorical": True,
            "cat_missing_fill": True,
            "weighted_ensemble": True,
        },
        "ablation_remove_catboost": {
            "use_xgb": True,
            "use_cat": False,
            "use_tabpfn": use_tabpfn,
            "xgb_label_encoding": True,
            "xgb_missing_fill": True,
            "cat_native_categorical": True,
            "cat_missing_fill": True,
            "weighted_ensemble": True,
        },
        "ablation_remove_xgboost": {
            "use_xgb": False,
            "use_cat": True,
            "use_tabpfn": use_tabpfn,
            "xgb_label_encoding": True,
            "xgb_missing_fill": True,
            "cat_native_categorical": True,
            "cat_missing_fill": True,
            "weighted_ensemble": True,
        },
        "ablation_equal_weights_instead_of_auc_weights": {
            "use_xgb": True,
            "use_cat": True,
            "use_tabpfn": use_tabpfn,
            "xgb_label_encoding": True,
            "xgb_missing_fill": True,
            "cat_native_categorical": True,
            "cat_missing_fill": True,
            "weighted_ensemble": False,
        },
        "ablation_disable_xgb_label_encoding_strategy": {
            "use_xgb": True,
            "use_cat": True,
            "use_tabpfn": use_tabpfn,
            "xgb_label_encoding": False,
            "xgb_missing_fill": True,
            "cat_native_categorical": True,
            "cat_missing_fill": True,
            "weighted_ensemble": True,
        },
        "ablation_disable_missing_value_fill": {
            "use_xgb": True,
            "use_cat": True,
            "use_tabpfn": False,
            "xgb_label_encoding": True,
            "xgb_missing_fill": False,
            "cat_native_categorical": True,
            "cat_missing_fill": False,
            "weighted_ensemble": True,
        },
        "ablation_catboost_without_native_categoricals": {
            "use_xgb": True,
            "use_cat": True,
            "use_tabpfn": use_tabpfn,
            "xgb_label_encoding": True,
            "xgb_missing_fill": True,
            "cat_native_categorical": False,
            "cat_missing_fill": True,
            "weighted_ensemble": True,
        },
    }

    results = {}
    baseline_score, baseline_parts = evaluate_pipeline(train, target_col, id_col, ablations["baseline_full_pipeline"])
    results["baseline_full_pipeline"] = baseline_score
    print(f"baseline_full_pipeline | ensemble_auc={baseline_score:.6f} | part_scores={baseline_parts}")

    for name, config in ablations.items():
        if name == "baseline_full_pipeline":
            continue
        score, parts = evaluate_pipeline(train, target_col, id_col, config)
        results[name] = score
        delta = score - baseline_score
        print(f"{name} | ensemble_auc={score:.6f} | delta_vs_baseline={delta:+.6f} | part_scores={parts}")

    contribution = {}
    for name, score in results.items():
        if name == "baseline_full_pipeline":
            continue
        contribution[name] = baseline_score - score

    most_important = max(contribution, key=contribution.get)
    print(f"most_contributing_part={most_important} | auc_drop={contribution[most_important]:.6f}")


if __name__ == "__main__":
    main()