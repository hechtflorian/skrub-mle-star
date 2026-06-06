
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

# Model choice: CatBoost if available, otherwise strong sklearn fallback.
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

final_validation_score = mean_squared_error(y_val, val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
