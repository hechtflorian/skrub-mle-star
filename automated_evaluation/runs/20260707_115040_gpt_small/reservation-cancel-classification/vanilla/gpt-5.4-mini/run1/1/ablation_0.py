
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
TARGET_COL = "booking_status"
ID_COL = "id"

def add_features(df):
    df = df.copy()

    # Basic time features
    if "arrival_month" in df.columns:
        df["is_peak_season"] = df["arrival_month"].isin([6, 7, 8, 12]).astype(int)
    if {"arrival_year", "arrival_month"}.issubset(df.columns):
        df["year_month"] = df["arrival_year"].astype(str) + "_" + df["arrival_month"].astype(str)

    # Stay length / demand related
    if {"no_of_weekend_nights", "no_of_week_nights"}.issubset(df.columns):
        df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]

    if {"no_of_adults", "no_of_children"}.issubset(df.columns):
        df["total_guests"] = df["no_of_adults"] + df["no_of_children"]

    if {"avg_price_per_room", "total_nights"}.issubset(df.columns):
        df["price_per_night"] = df["avg_price_per_room"] / (df["total_nights"] + 1)

    if {"lead_time", "total_nights"}.issubset(df.columns):
        df["lead_time_per_night"] = df["lead_time"] / (df["total_nights"] + 1)

    if {"no_of_special_requests", "required_car_parking_space"}.issubset(df.columns):
        df["special_requests_plus_parking"] = df["no_of_special_requests"] + df["required_car_parking_space"]

    if {"repeated_guest", "no_of_previous_cancellations"}.issubset(df.columns):
        df["previous_issue_flag"] = (
            (df["repeated_guest"] == 1) | (df["no_of_previous_cancellations"] > 0)
        ).astype(int)

    return df

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = add_features(train_df)
test_df = add_features(test_df)

y = train_df[TARGET_COL].astype(int)
train_ids = train_df[ID_COL].copy()
test_ids = test_df[ID_COL].copy()

X = train_df.drop(columns=[TARGET_COL])
X_test = test_df.copy()

# Drop ID from features
if ID_COL in X.columns:
    X = X.drop(columns=[ID_COL])
if ID_COL in X_test.columns:
    X_test = X_test.drop(columns=[ID_COL])

# Identify categorical features: original low-cardinality integer features + created string features
cat_cols = []
for col in X.columns:
    if X[col].dtype == "object":
        cat_cols.append(col)
    elif pd.api.types.is_integer_dtype(X[col]) and X[col].nunique(dropna=False) <= 20:
        cat_cols.append(col)

cat_cols = sorted(list(set(cat_cols)))

# Prepare train/validation split
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def fit_lgbm(X_tr, y_tr, X_va, y_va, cat_cols, use_class_weight=True):
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()

    for c in cat_cols:
        if c in X_tr_lgb.columns:
            X_tr_lgb[c] = X_tr_lgb[c].astype("category")
            X_va_lgb[c] = X_va_lgb[c].astype("category")

    params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,          # keep subsampling
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        min_child_samples=20,
        reg_lambda=1.0,
    )
    if use_class_weight:
        params["class_weight"] = "balanced"

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    pred = model.predict_proba(X_va_lgb)[:, 1]
    auc = roc_auc_score(y_va, pred)
    return model, auc

def fit_catboost(X_tr, y_tr, X_va, y_va, cat_cols, use_class_weights=True):
    params = dict(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100
    )
    if use_class_weights:
        params["auto_class_weights"] = "Balanced"

    model = CatBoostClassifier(**params)
    cat_features = [c for c in cat_cols if c in X_tr.columns]
    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_features,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, pred)
    return model, auc

# Train models for final prediction
lgb_model, lgb_auc = fit_lgbm(X_tr, y_tr, X_va, y_va, cat_cols, use_class_weight=True)
cat_model, cat_auc = fit_catboost(X_tr, y_tr, X_va, y_va, cat_cols, use_class_weights=True)

# Validation ensemble
X_va_lgb = X_va.copy()
X_va_test_lgb = X_test.copy()
for c in cat_cols:
    if c in X_va_lgb.columns:
        X_va_lgb[c] = X_va_lgb[c].astype("category")
    if c in X_va_test_lgb.columns:
        X_va_test_lgb[c] = X_va_test_lgb[c].astype("category")

lgb_pred_va = lgb_model.predict_proba(X_va_lgb)[:, 1]
cat_pred_va = cat_model.predict_proba(X_va)[:, 1]
ens_pred_va = 0.5 * lgb_pred_va + 0.5 * cat_pred_va
final_validation_score = roc_auc_score(y_va, ens_pred_va)

print(f"Final Validation Performance: {final_validation_score:.6f}")
print(f"LightGBM Validation AUC: {lgb_auc:.6f}")
print(f"CatBoost Validation AUC: {cat_auc:.6f}")

# Refit on full training data for test prediction
X_full = X.copy()
X_test_full = X_test.copy()

# LightGBM full fit
X_full_lgb = X_full.copy()
X_test_lgb = X_test_full.copy()
for c in cat_cols:
    if c in X_full_lgb.columns:
        X_full_lgb[c] = X_full_lgb[c].astype("category")
    if c in X_test_lgb.columns:
        X_test_lgb[c] = X_test_lgb[c].astype("category")

lgb_full = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    min_child_samples=20,
    reg_lambda=1.0,
    class_weight="balanced"
)
lgb_full.fit(
    X_full_lgb,
    y,
    eval_metric="auc",
    callbacks=[lgb.log_evaluation(0)]
)
lgb_test_pred = lgb_full.predict_proba(X_test_lgb)[:, 1]

# CatBoost full fit
cat_full = CatBoostClassifier(
    iterations=cat_model.get_param("iterations") if hasattr(cat_model, "get_param") else 2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    auto_class_weights="Balanced"
)
cat_features_full = [c for c in cat_cols if c in X_full.columns]
cat_full.fit(
    X_full,
    y,
    cat_features=cat_features_full
)
cat_test_pred = cat_full.predict_proba(X_test_full)[:, 1]

# Ensemble test predictions
test_pred = 0.5 * lgb_test_pred + 0.5 * cat_test_pred
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: (test_pred >= 0.5).astype(int)
})

submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
print("Saved submission.csv")
