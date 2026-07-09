
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_map = {"Introvert": 0, "Extrovert": 1}
inv_target_map = {0: "Introvert", 1: "Extrovert"}

y = train["Personality"].map(target_map).astype(int)
X = train.drop(columns=["Personality"]).copy()
X_test = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]
for col in num_cols:
    med = X[col].median()
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

X["id"] = X["id"].fillna(-1)
X_test["id"] = X_test["id"].fillna(-1)

X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# -------- CatBoost branch --------
cat_cols_cb = [c for c in X_tr_raw.columns if X_tr_raw[c].dtype == "object"]
train_pool = Pool(X_tr_raw, y_tr, cat_features=cat_cols_cb)
val_pool = Pool(X_val_raw, y_val, cat_features=cat_cols_cb)
test_pool = Pool(X_test, cat_features=cat_cols_cb)

cb_model = CatBoostClassifier(
    iterations=1500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=SEED,
    verbose=200
)
cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

cb_val_pred = cb_model.predict(X_val_raw).astype(int).ravel()
cb_val_acc = accuracy_score(y_val, cb_val_pred)
cb_test_pred_proba = cb_model.predict_proba(test_pool)[:, 1]

# -------- LightGBM branch --------
X_lgb = pd.get_dummies(X, drop_first=True)
X_test_lgb = pd.get_dummies(X_test, drop_first=True)
X_lgb, X_test_lgb = X_lgb.align(X_test_lgb, join="left", axis=1, fill_value=0)

X_tr_lgb = X_lgb.loc[X_tr_raw.index]
X_val_lgb = X_lgb.loc[X_val_raw.index]

lgb_model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=SEED,
    subsample=0.9,
    colsample_bytree=0.9
)
lgb_model.fit(X_tr_lgb, y_tr)

lgb_val_pred = lgb_model.predict(X_val_lgb)
lgb_val_acc = accuracy_score(y_val, lgb_val_pred)
lgb_test_pred_proba = lgb_model.predict_proba(X_test_lgb)[:, 1]

# -------- Simple ensemble --------
ensemble_val_proba = 0.5 * cb_model.predict_proba(X_val_raw)[:, 1] + 0.5 * lgb_model.predict_proba(X_val_lgb)[:, 1]
ensemble_val_pred = (ensemble_val_proba >= 0.5).astype(int)
final_val_acc = accuracy_score(y_val, ensemble_val_pred)

ensemble_test_proba = 0.5 * cb_test_pred_proba + 0.5 * lgb_test_pred_proba
test_pred = (ensemble_test_proba >= 0.5).astype(int)

print(f"CatBoost Validation Performance: {cb_val_acc}")
print(f"LightGBM Validation Performance: {lgb_val_acc}")
print(f"Final Validation Performance: {final_val_acc}")

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(test_pred).map(inv_target_map)
})
submission.to_csv("submission.csv", index=False)
