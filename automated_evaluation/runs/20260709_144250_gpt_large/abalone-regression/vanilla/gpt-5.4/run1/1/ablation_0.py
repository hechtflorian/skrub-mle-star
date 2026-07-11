
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
train = pd.read_csv(train_path)

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
if len(missing_train) > 0:
    raise ValueError(f"Missing columns in train.csv: {missing_train}")

df = train[features + [target_col]].copy()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).clip(lower=0)

for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in cat_features:
    df[col] = df[col].fillna("Unknown").astype(str)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

X_tr = train_df[features].copy()
y_tr = train_df[target_col].copy()
X_va = val_df[features].copy()
y_va = val_df[target_col].copy()

def build_cat_model():
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_seed=SEED,
        verbose=False,
    )

def build_hgb_pipeline():
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", hgb_base_model),
        ]
    )

def rmsle_score(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def run_config(name, use_cat=True, use_hgb=True, hgb_log_target=True, ensemble_weights=None):
    preds = []

    if use_cat:
        cat_model = build_cat_model()
        cat_model.fit(
            X_tr,
            y_tr,
            cat_features=cat_features,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=200,
        )
        cat_val_pred = np.clip(cat_model.predict(X_va), 0, None)
        preds.append(("cat", cat_val_pred))

    if use_hgb:
        hgb_model = build_hgb_pipeline()
        if hgb_log_target:
            y_fit = np.log1p(y_tr.values)
            hgb_model.fit(X_tr, y_fit)
            hgb_val_pred = np.expm1(hgb_model.predict(X_va))
        else:
            hgb_model.fit(X_tr, y_tr.values)
            hgb_val_pred = hgb_model.predict(X_va)
        hgb_val_pred = np.clip(hgb_val_pred, 0, None)
        preds.append(("hgb", hgb_val_pred))

    if len(preds) == 0:
        raise ValueError("At least one model must be enabled.")

    if ensemble_weights is None:
        ensemble_weights = [1.0 / len(preds)] * len(preds)

    val_pred = np.zeros(len(X_va), dtype=float)
    for w, (_, p) in zip(ensemble_weights, preds):
        val_pred += w * p

    score = rmsle_score(y_va, val_pred)
    print(f"{name}: RMSLE={score:.6f}")
    return score

results = {}
results["baseline_full_ensemble"] = run_config(
    name="baseline_full_ensemble",
    use_cat=True,
    use_hgb=True,
    hgb_log_target=True,
    ensemble_weights=[0.5, 0.5],
)

results["ablation_no_catboost"] = run_config(
    name="ablation_no_catboost",
    use_cat=False,
    use_hgb=True,
    hgb_log_target=True,
)

results["ablation_no_hgb"] = run_config(
    name="ablation_no_hgb",
    use_cat=True,
    use_hgb=False,
)

results["ablation_hgb_without_log_target"] = run_config(
    name="ablation_hgb_without_log_target",
    use_cat=True,
    use_hgb=True,
    hgb_log_target=False,
    ensemble_weights=[0.5, 0.5],
)

baseline = results["baseline_full_ensemble"]
drops = {k: v - baseline for k, v in results.items() if k != "baseline_full_ensemble"}

print("\nAblation impact vs baseline:")
for name, drop in sorted(drops.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: RMSLE change = {drop:+.6f}")

worst_ablation = max(drops, key=drops.get)
if worst_ablation == "ablation_no_catboost":
    contributor = "CatBoost model"
elif worst_ablation == "ablation_no_hgb":
    contributor = "HistGradientBoosting model"
elif worst_ablation == "ablation_hgb_without_log_target":
    contributor = "log1p target transform for HGB"
else:
    contributor = worst_ablation

print(f"\nLargest performance contributor: {contributor} ({worst_ablation})")
