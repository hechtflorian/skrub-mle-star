
import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "NObeyesdad"
y = train[target]
X = train.drop(columns=[target])

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

test_ids = test["id"].copy()

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Preprocessing for LightGBM ----------
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_tr_lgb = X_tr.copy()
X_val_lgb = X_val.copy()
test_lgb = test.copy()

X_tr_lgb[cat_cols] = enc.fit_transform(X_tr_lgb[cat_cols])
X_val_lgb[cat_cols] = enc.transform(X_val_lgb[cat_cols])
test_lgb[cat_cols] = enc.transform(test_lgb[cat_cols])

X_tr_lgb = X_tr_lgb.drop(columns=["id"])
X_val_lgb = X_val_lgb.drop(columns=["id"])
test_lgb_features = test_lgb.drop(columns=["id"])

# ---------- CatBoost can use raw categorical features ----------
X_tr_cb = X_tr.copy()
X_val_cb = X_val.copy()
test_cb = test.copy()

# Drop id from CatBoost features
X_tr_cb = X_tr_cb.drop(columns=["id"])
X_val_cb = X_val_cb.drop(columns=["id"])
test_cb_features = test_cb.drop(columns=["id"])

cat_features_cb = [X_tr_cb.columns.get_loc(c) for c in cat_cols if c in X_tr_cb.columns]

# Optional device info
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------- Model 1: CatBoost ----------
cat_model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=2000,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=200
)

cat_model.fit(
    X_tr_cb,
    y_tr,
    cat_features=cat_features_cb,
    eval_set=(X_val_cb, y_val),
    use_best_model=True
)

# ---------- Model 2: LightGBM ----------
lgb_model = LGBMClassifier(
    objective="multiclass",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    random_state=42,
    class_weight=None
)

lgb_model.fit(
    X_tr_lgb,
    y_tr,
    eval_set=[(X_val_lgb, y_val)],
    eval_metric="multi_logloss"
)

# ---------- Validation ensemble ----------
cat_val_proba = cat_model.predict_proba(X_val_cb)
lgb_val_proba = lgb_model.predict_proba(X_val_lgb)

ensemble_val_proba = 0.5 * cat_val_proba + 0.5 * lgb_val_proba
ensemble_val_pred = np.array(cat_model.classes_)[np.argmax(ensemble_val_proba, axis=1)]

final_validation_score = accuracy_score(y_val, ensemble_val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# ---------- Test ensemble ----------
cat_test_proba = cat_model.predict_proba(test_cb_features)
lgb_test_proba = lgb_model.predict_proba(test_lgb_features)

ensemble_test_proba = 0.5 * cat_test_proba + 0.5 * lgb_test_proba
test_pred = np.array(cat_model.classes_)[np.argmax(ensemble_test_proba, axis=1)]

submission = pd.DataFrame({
    "id": test_ids,
    "NObeyesdad": test_pred
})
submission.to_csv("submission.csv", index=False)
