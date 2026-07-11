
import os
import random
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_package("catboost")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import sklearn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "./input"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

features = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Whole weight.1",
    "Whole weight.2",
    "Shell weight",
]
cat_features = ["Sex"]
num_features = [c for c in features if c not in cat_features]
target_col = "Rings"

missing_train = [c for c in features + [target_col] if c not in train.columns]
missing_test = [c for c in ["id"] + features if c not in test.columns]
if len(missing_train) > 0:
    raise ValueError(f"Missing columns in train.csv: {missing_train}")
if len(missing_test) > 0:
    raise ValueError(f"Missing columns in test.csv: {missing_test}")

df = train[features + [target_col]].copy()
test_df = test[["id"] + features].copy()

df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).clip(lower=0)

for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

for col in cat_features:
    df[col] = df[col].fillna("Unknown").astype(str)
    test_df[col] = test_df[col].fillna("Unknown").astype(str)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

X_tr = train_df[features].copy()
y_tr = train_df[target_col].copy()
X_va = val_df[features].copy()
y_va = val_df[target_col].copy()
X_test = test_df[features].copy()

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=SEED,
    verbose=200,
)

cat_model.fit(
    X_tr,
    y_tr,
    cat_features=cat_features,
    eval_set=(X_va, y_va),
    use_best_model=True,
    early_stopping_rounds=200,
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

hgb_base_model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.03,
    max_iter=600,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=SEED,
)

hgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", hgb_base_model),
    ]
)

y_tr_log = np.log1p(y_tr.values)
hgb_model.fit(X_tr, y_tr_log)


cat_val_pred = cat_model.predict(X_va)
cat_val_pred = np.clip(cat_val_pred, 0, None)

hgb_val_pred_log = hgb_model.predict(X_va)
hgb_val_pred = np.expm1(hgb_val_pred_log)
hgb_val_pred = np.clip(hgb_val_pred, 0, None)

blend_weights = [0.5, 0.6, 0.7, 0.8]
best_blend_weight = None
best_blend_score = np.inf

for w_cat in blend_weights:
    val_pred_candidate = w_cat * cat_val_pred + (1.0 - w_cat) * hgb_val_pred
    val_pred_candidate = np.clip(val_pred_candidate, 0, None)
    score = np.sqrt(mean_squared_log_error(y_va, val_pred_candidate))
    if score < best_blend_score:
        best_blend_score = score
        best_blend_weight = w_cat

val_pred = best_blend_weight * cat_val_pred + (1.0 - best_blend_weight) * hgb_val_pred
val_pred = np.clip(val_pred, 0, None)

final_validation_score = np.sqrt(mean_squared_log_error(y_va, val_pred))

best_iter = cat_model.get_best_iteration()
if best_iter is None or best_iter < 0:
    best_iter = 3000 - 1

full_cat_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=max(best_iter + 200, 1),
    learning_rate=0.03,
    depth=9,
    l2_leaf_reg=3,
    random_seed=SEED,
    verbose=False,
)

full_cat_model.fit(
    df[features],
    df[target_col],
    cat_features=cat_features,
)

full_hgb_model = clone(hgb_model)
y_full_log = np.log1p(df[target_col].values)
full_hgb_model.fit(df[features], y_full_log)

cat_test_pred = full_cat_model.predict(X_test)
cat_test_pred = np.clip(cat_test_pred, 0, None)

hgb_test_pred_log = full_hgb_model.predict(X_test)
hgb_test_pred = np.expm1(hgb_test_pred_log)
hgb_test_pred = np.clip(hgb_test_pred, 0, None)

test_pred = best_blend_weight * cat_test_pred + (1.0 - best_blend_weight) * hgb_test_pred
test_pred = np.clip(test_pred, 0, None)


submission = pd.DataFrame({
    "id": test_df["id"],
    "Rings": test_pred
})
submission.to_csv("submission_catboost_hgb_ensemble.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
