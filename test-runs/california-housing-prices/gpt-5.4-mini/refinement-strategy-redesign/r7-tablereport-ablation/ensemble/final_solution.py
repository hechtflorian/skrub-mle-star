
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

# Holdout validation for performance reporting
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def rank_percentile(arr):
    s = pd.Series(np.asarray(arr).reshape(-1))
    return s.rank(method="average", pct=True).to_numpy()

def softmax_neg_rmse(rmses, temperature=1.0):
    vals = np.asarray(rmses, dtype=float)
    z = -vals / max(temperature, 1e-8)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def refit_and_predict_full(model_name, X_full, y_full, X_val_local, X_test_local):
    if model_name == "catboost":
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
        model.fit(X_full, y_full, verbose=False)
        val_pred = model.predict(X_val_local)
        test_pred = model.predict(X_test_local)
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            random_state=42,
        )
        model.fit(X_full, y_full)
        val_pred = model.predict(X_val_local)
        test_pred = model.predict(X_test_local)
    return val_pred, test_pred

# Base pipelines exactly as they are for validation estimation
if USE_CATBOOST:
    base1 = CatBoostRegressor(
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
    base1.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )
    val_pred_1 = base1.predict(X_val)
else:
    base1 = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        random_state=42,
    )
    base1.fit(X_train, y_train)
    val_pred_1 = base1.predict(X_val)

base2 = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=500,
    random_state=42,
)
base2.fit(X_train, y_train)
val_pred_2 = base2.predict(X_val)

rmse_1 = mean_squared_error(y_val, val_pred_1) ** 0.5
rmse_2 = mean_squared_error(y_val, val_pred_2) ** 0.5

# Refit each base model on full training data once, then predict test/validation
if USE_CATBOOST:
    val_pred_1_full, test_pred_1 = refit_and_predict_full("catboost", X, y, X_val, test_df)
else:
    val_pred_1_full, test_pred_1 = refit_and_predict_full("hgb", X, y, X_val, test_df)

val_pred_2_full, test_pred_2 = refit_and_predict_full("hgb", X, y, X_val, test_df)

# Rank-based normalization on validation/test predictions
val_rank_1 = rank_percentile(val_pred_1)
val_rank_2 = rank_percentile(val_pred_2)
test_rank_1 = rank_percentile(test_pred_1)
test_rank_2 = rank_percentile(test_pred_2)

# Softmax weights from validation RMSE
weights = softmax_neg_rmse([rmse_1, rmse_2], temperature=1.0)
w1, w2 = float(weights[0]), float(weights[1])

# Blend ranked predictions
val_rank_blend = w1 * val_rank_1 + w2 * val_rank_2
test_rank_blend = w1 * test_rank_1 + w2 * test_rank_2

# Optional robust back-transform using validation target statistics
y_median = float(np.median(y_val))
y_q1 = float(np.percentile(y_val, 25))
y_q3 = float(np.percentile(y_val, 75))
y_iqr = max(y_q3 - y_q1, 1e-8)

val_blend_scaled = y_median + (val_rank_blend - 0.5) * y_iqr
test_blend_scaled = y_median + (test_rank_blend - 0.5) * y_iqr

# Agreement gate / disagreement-based shrinkage on original-scale test predictions
centered_pred = 0.5 * (test_pred_1 + test_pred_2)
disagreement = np.abs(test_pred_1 - test_pred_2)
q95_disagreement = float(np.percentile(np.abs(val_pred_1 - val_pred_2), 95))
q95_disagreement = max(q95_disagreement, 1e-8)
shrink = np.clip(disagreement / q95_disagreement, 0.0, 1.0)

# Use weighted average directly when close; shrink toward validation median when far apart
weighted_direct = w1 * test_pred_1 + w2 * test_pred_2
gate_blend = weighted_direct * (1.0 - 0.15 * shrink) + y_median * (0.15 * shrink)

# Combine rank-based blend and agreement-gated original-scale blend
test_pred = 0.5 * test_blend_scaled + 0.5 * gate_blend

# Validation prediction from the same final ensemble logic
val_centered_pred = 0.5 * (val_pred_1 + val_pred_2)
val_disagreement = np.abs(val_pred_1 - val_pred_2)
val_shrink = np.clip(val_disagreement / q95_disagreement, 0.0, 1.0)
val_weighted_direct = w1 * val_pred_1 + w2 * val_pred_2
val_gate_blend = val_weighted_direct * (1.0 - 0.15 * val_shrink) + y_median * (0.15 * val_shrink)
val_final_pred = 0.5 * (y_median + (val_rank_blend - 0.5) * y_iqr) + 0.5 * val_gate_blend

final_validation_score = mean_squared_error(y_val, val_final_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("./final/submission.csv", index=False)
print(submission.head().to_string(index=False))
