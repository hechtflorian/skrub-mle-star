
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
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
except Exception:
    skrub = None

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

# Same holdout split for both models so validation predictions align row-by-row
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def fit_solution_1(X_train, y_train, X_val, y_val, test_df):
    if USE_CATBOOST:
        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            subsample=0.8,
            bagging_temperature=0.5,
            l2_leaf_reg=3.0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        val_pred = model.predict(X_val)
        test_pred = model.predict(test_df)
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(test_df)
    return np.asarray(val_pred), np.asarray(test_pred)

def fit_solution_2(X_train, y_train, X_val, y_val, test_df):
    # Minimal independent base model: keep a separate sklearn fallback pipeline.
    # If skrub is available, stay DataOps-friendly via TableVectorizer, otherwise use a direct HGB baseline.
    if skrub is not None:
        try:
            from skrub import TableVectorizer
            vec = TableVectorizer()
            X_train_vec = vec.fit_transform(X_train)
            X_val_vec = vec.transform(X_val)
            X_test_vec = vec.transform(test_df)

            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=700,
                random_state=42,
            )
            model.fit(X_train_vec, y_train)
            val_pred = model.predict(X_val_vec)
            test_pred = model.predict(X_test_vec)
            return np.asarray(val_pred), np.asarray(test_pred)
        except Exception:
            pass

    model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_depth=7,
        max_iter=700,
        random_state=42,
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(test_df)
    return np.asarray(val_pred), np.asarray(test_pred)

# Base model predictions
val_pred_1, test_pred_1 = fit_solution_1(X_train, y_train, X_val, y_val, test_df)
val_pred_2, test_pred_2 = fit_solution_2(X_train, y_train, X_val, y_val, test_df)

# Base validation metrics
rmse_1 = mean_squared_error(y_val, val_pred_1) ** 0.5
rmse_2 = mean_squared_error(y_val, val_pred_2) ** 0.5

# Simple validation-tuned weighted average
eps = 1e-8
w1 = 1.0 / (rmse_1 + eps)
w2 = 1.0 / (rmse_2 + eps)
w_sum = w1 + w2
weight_1 = w1 / w_sum
weight_2 = w2 / w_sum

blend_val_weighted = weight_1 * val_pred_1 + weight_2 * val_pred_2
blend_test_weighted = weight_1 * test_pred_1 + weight_2 * test_pred_2
blend_rmse_weighted = mean_squared_error(y_val, blend_val_weighted) ** 0.5

# Optional very small meta-model on validation predictions only
# Use a conservative linear model; choose it only if it improves validation RMSE.
meta_X_val = np.column_stack([val_pred_1, val_pred_2])
meta_X_test = np.column_stack([test_pred_1, test_pred_2])

# Ridge is safer than unconstrained OLS in case predictions are highly correlated.
meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(meta_X_val, y_val)
meta_val_pred = meta_model.predict(meta_X_val)
meta_test_pred = meta_model.predict(meta_X_test)
meta_rmse = mean_squared_error(y_val, meta_val_pred) ** 0.5

# Select the best simple blend by validation RMSE
if meta_rmse < blend_rmse_weighted:
    final_validation_score = meta_rmse
    final_test_pred = meta_test_pred
else:
    final_validation_score = blend_rmse_weighted
    final_test_pred = blend_test_weighted

print(f"Solution1 Validation RMSE: {rmse_1}")
print(f"Solution2 Validation RMSE: {rmse_2}")
print(f"Weighted Blend Validation RMSE: {blend_rmse_weighted}")
print(f"Meta Blend Validation RMSE: {meta_rmse}")
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({TARGET_COL: final_test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
