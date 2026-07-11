
import os
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = train["Attrition"].astype(int)
X = train.drop(columns=["Attrition"]).copy()
T = test.copy()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

for c in cat_cols:
    X[c] = X[c].astype(str).fillna("NA")
    T[c] = T[c].astype(str).fillna("NA")

for c in num_cols:
    median_val = X[c].median()
    X[c] = X[c].fillna(median_val)
    T[c] = T[c].fillna(median_val)

all_df = pd.concat([X, T], axis=0, ignore_index=True)
all_df = pd.get_dummies(all_df, dummy_na=True)

X_enc = all_df.iloc[:len(X)].copy()
T_enc = all_df.iloc[len(X):].copy()

X_train_enc, X_val_enc, y_train, y_val = train_test_split(
    X_enc, y, test_size=0.2, random_state=SEED, stratify=y
)

X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=SEED,
    subsample=0.9,
    colsample_bytree=0.9,
)

lgbm_model.fit(
    X_train_enc,
    y_train,
    eval_set=[(X_val_enc, y_val)],
    eval_metric="auc",
)

lgbm_val_pred = lgbm_model.predict_proba(X_val_enc)[:, 1]
lgbm_test_pred = lgbm_model.predict_proba(T_enc)[:, 1]

train_pool = Pool(X_train_cat, y_train_cat, cat_features=cat_cols)
val_pool = Pool(X_val_cat, y_val_cat, cat_features=cat_cols)
test_pool = Pool(T, cat_features=cat_cols)

cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=False
)

cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

cat_val_pred = cat_model.predict_proba(val_pool)[:, 1]
cat_test_pred = cat_model.predict_proba(test_pool)[:, 1]

lgbm_score = roc_auc_score(y_val, lgbm_val_pred)
cat_score = roc_auc_score(y_val_cat, cat_val_pred)

val_pred_ensemble = (lgbm_val_pred + cat_val_pred) / 2.0
final_validation_score = roc_auc_score(y_val, val_pred_ensemble)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = (lgbm_test_pred + cat_test_pred) / 2.0

id_col = "EmployeeNumber" if "EmployeeNumber" in test.columns else ("id" if "id" in test.columns else None)
submission_ids = test[id_col] if id_col is not None else np.arange(len(test))

submission = pd.DataFrame({
    "EmployeeNumber": submission_ids,
    "Attrition": test_pred
})
submission.to_csv("submission.csv", index=False)
