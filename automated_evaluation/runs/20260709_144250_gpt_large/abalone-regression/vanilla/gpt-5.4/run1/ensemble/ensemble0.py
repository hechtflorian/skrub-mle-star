
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

y_va_log = np.log1p(y_va.values)
cat_val_log = np.log1p(cat_val_pred)

if np.std(cat_val_log) < 1e-12:
    a = 1.0
    b = float(np.mean(y_va_log) - np.mean(cat_val_log))
else:
    a, b = np.polyfit(cat_val_log, y_va_log, 1)

cat_cal_val_log = a * cat_val_log + b
cat_cal_val_pred = np.expm1(cat_cal_val_log)
cat_cal_val_pred = np.clip(cat_cal_val_pred, 0, None)

candidate_scores = []
candidate_preds = {}
candidate_rules = {}

score_cat_cal = np.sqrt(mean_squared_log_error(y_va, cat_cal_val_pred))
candidate_scores.append(("cat_calibrated", score_cat_cal))
candidate_preds["cat_calibrated"] = cat_cal_val_pred
candidate_rules["cat_calibrated"] = {"type": "cat_calibrated"}

score_hgb = np.sqrt(mean_squared_log_error(y_va, hgb_val_pred))
candidate_scores.append(("hgb", score_hgb))
candidate_preds["hgb"] = hgb_val_pred
candidate_rules["hgb"] = {"type": "hgb"}

raw_avg_val_pred = 0.5 * cat_val_pred + 0.5 * hgb_val_pred
raw_avg_val_pred = np.clip(raw_avg_val_pred, 0, None)
score_raw_avg = np.sqrt(mean_squared_log_error(y_va, raw_avg_val_pred))
candidate_scores.append(("raw_avg_0.5", score_raw_avg))
candidate_preds["raw_avg_0.5"] = raw_avg_val_pred
candidate_rules["raw_avg_0.5"] = {"type": "raw_avg", "w_cat": 0.5}

log_blend_weights = [0.2, 0.35, 0.5, 0.65, 0.8]
for w_cat in log_blend_weights:
    blend_log = w_cat * cat_cal_val_log + (1.0 - w_cat) * hgb_val_pred_log
    blend_pred = np.expm1(blend_log)
    blend_pred = np.clip(blend_pred, 0, None)
    score_blend = np.sqrt(mean_squared_log_error(y_va, blend_pred))
    name = f"log_blend_{w_cat}"
    candidate_scores.append((name, score_blend))
    candidate_preds[name] = blend_pred
    candidate_rules[name] = {"type": "log_blend", "w_cat": w_cat}

best_name, best_score = min(candidate_scores, key=lambda x: x[1])
val_pred = candidate_preds[best_name]
final_validation_score = best_score
best_rule = candidate_rules[best_name]

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

cat_test_log = np.log1p(cat_test_pred)
cat_cal_test_log = a * cat_test_log + b
cat_cal_test_pred = np.expm1(cat_cal_test_log)
cat_cal_test_pred = np.clip(cat_cal_test_pred, 0, None)

if best_rule["type"] == "cat_calibrated":
    test_pred = cat_cal_test_pred
elif best_rule["type"] == "hgb":
    test_pred = hgb_test_pred
elif best_rule["type"] == "raw_avg":
    test_pred = best_rule["w_cat"] * cat_test_pred + (1.0 - best_rule["w_cat"]) * hgb_test_pred
elif best_rule["type"] == "log_blend":
    blend_test_log = best_rule["w_cat"] * cat_cal_test_log + (1.0 - best_rule["w_cat"]) * hgb_test_pred_log
    test_pred = np.expm1(blend_test_log)
else:
    test_pred = 0.5 * cat_cal_test_pred + 0.5 * hgb_test_pred

test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "id": test_df["id"],
    "Rings": test_pred
})
submission.to_csv("submission_catboost_hgb_ensemble.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
