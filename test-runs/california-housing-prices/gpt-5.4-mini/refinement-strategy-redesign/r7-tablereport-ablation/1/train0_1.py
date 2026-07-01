
import os
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

# Install missing dependencies if needed.
try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])
    import lightgbm as lgb
    USE_LIGHTGBM = True

try:
    import catboost
    from catboost import CatBoostRegressor
    USE_CATBOOST = True
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostRegressor
    USE_CATBOOST = True

try:
    import skrub
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "skrub"])
    import skrub

RANDOM_STATE = 42
TARGET_COL = "median_house_value"
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")

data = skrub.var("data", train_df)
X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y = data[TARGET_COL].skb.mark_as_y()

# Shared preprocessing in the DataOps graph
imputer = SimpleImputer(strategy="median")
X_imputed = X.skb.apply(imputer)

# Model 1: LightGBM (reference solution)
lgb_model = lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
lgb_pred = X_imputed.skb.apply(lgb_model, y=y)
lgb_learner = lgb_pred.skb.make_learner(fitted=True)

# Model 2: CatBoost (base solution)
cat_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=RANDOM_STATE,
    verbose=False,
    subsample=0.8,
    bagging_temperature=0.5,
    l2_leaf_reg=3.0,
)
cat_pred = X_imputed.skb.apply(cat_model, y=y)
cat_learner = cat_pred.skb.make_learner(fitted=True)

# Validation split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE
)
train_split = train_df.iloc[train_idx].copy()
valid_split = train_df.iloc[valid_idx].copy()

# Fit LightGBM on split
split_data_lgb = skrub.var("data", train_split)
split_X_lgb = split_data_lgb.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
split_y_lgb = split_data_lgb[TARGET_COL].skb.mark_as_y()
split_X_lgb_imputed = split_X_lgb.skb.apply(SimpleImputer(strategy="median"))
split_lgb_pred = split_X_lgb_imputed.skb.apply(
    lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    y=split_y_lgb,
)
split_lgb_learner = split_lgb_pred.skb.make_learner(fitted=True)
valid_lgb_preds = split_lgb_learner.predict({"data": valid_split})

# Fit CatBoost on split
split_data_cat = skrub.var("data", train_split)
split_X_cat = split_data_cat.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
split_y_cat = split_data_cat[TARGET_COL].skb.mark_as_y()
split_X_cat_imputed = split_X_cat.skb.apply(SimpleImputer(strategy="median"))
split_cat_pred = split_X_cat_imputed.skb.apply(
    CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        subsample=0.8,
        bagging_temperature=0.5,
        l2_leaf_reg=3.0,
    ),
    y=split_y_cat,
)
split_cat_learner = split_cat_pred.skb.make_learner(fitted=True)
valid_cat_preds = split_cat_learner.predict({"data": valid_split})

# Simple ensemble on validation
valid_preds = 0.5 * np.asarray(valid_lgb_preds) + 0.5 * np.asarray(valid_cat_preds)
final_validation_score = mean_squared_error(valid_split[TARGET_COL].values, valid_preds) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Predict on test set and ensemble
test_lgb_preds = lgb_learner.predict({"data": test_df})
test_cat_preds = cat_learner.predict({"data": test_df})
test_preds = 0.5 * np.asarray(test_lgb_preds) + 0.5 * np.asarray(test_cat_preds)

submission = pd.DataFrame({TARGET_COL: test_preds})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
