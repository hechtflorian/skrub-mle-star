

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

# Install missing dependency if needed, then import CatBoost.
try:
    from catboost import CatBoostRegressor
    USE_CATBOOST = True
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostRegressor
    USE_CATBOOST = True

try:
    import skrub
    from skrub import selectors as s
except Exception:
    skrub = None
    s = None

DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Basic cleanup
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")

X = train_df.drop(columns=[TARGET_COL], errors="ignore")
y = train_df[TARGET_COL].astype(float)

# Holdout validation for performance reporting
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Keep the strong baseline and compare against a single, fixed TableVectorizer path.
# This stays aligned with the ablation result: minimal encoding change, no manual feature crafting.
baseline_model_params = dict(
    learning_rate=0.05,
    max_depth=8,
    max_iter=500,
    random_state=42,
)

def fit_and_score_baseline(X_tr, y_tr, X_va, y_va, X_te):
    model = HistGradientBoostingRegressor(**baseline_model_params)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_va)
    test_pred = model.predict(X_te)
    rmse = mean_squared_error(y_va, val_pred) ** 0.5
    return rmse, val_pred, test_pred, model

def fit_and_score_vectorized(X_tr, y_tr, X_va, y_va, X_te):
    if skrub is None:
        return np.inf, None, None, None

    # Fixed, compact DataOps preprocessing:
    # - low-cardinality -> ToCategorical
    # - high-cardinality -> MinHashEncoder
    # This is the only encoding change we test, per ablation guidance.
    vectorizer = skrub.TableVectorizer(
        low_cardinality=skrub.ToCategorical(),
        high_cardinality=skrub.MinHashEncoder(),
    )

    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_va_vec = vectorizer.transform(X_va)
    X_te_vec = vectorizer.transform(X_te)

    model = HistGradientBoostingRegressor(**baseline_model_params)
    model.fit(X_tr_vec, y_tr)
    val_pred = model.predict(X_va_vec)
    test_pred = model.predict(X_te_vec)
    rmse = mean_squared_error(y_va, val_pred) ** 0.5
    return rmse, val_pred, test_pred, model

baseline_rmse, baseline_val_pred, baseline_test_pred, baseline_model = fit_and_score_baseline(
    X_train, y_train, X_val, y_val, test_df
)

vectorized_rmse, vectorized_val_pred, vectorized_test_pred, vectorized_model = fit_and_score_vectorized(
    X_train, y_train, X_val, y_val, test_df
)

if vectorized_rmse < baseline_rmse:
    final_validation_score = vectorized_rmse
    test_pred = vectorized_test_pred
    selected_variant = "tablevectorizer_hgb"
else:
    final_validation_score = baseline_rmse
    test_pred = baseline_test_pred
    selected_variant = "baseline_hgb"

print(f"Selected variant: {selected_variant}")
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
