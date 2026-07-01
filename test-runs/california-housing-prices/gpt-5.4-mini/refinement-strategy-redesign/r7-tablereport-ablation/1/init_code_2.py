
import os
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import skrub

RANDOM_STATE = 42
TARGET = "median_house_value"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

data = skrub.var("data", train_df)
X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

# LightGBM works well on numeric tabular data; keep the DataOps-first structure.
# Use a simple, robust preprocessing path: impute missing values, then train.
imputer = SimpleImputer(strategy="median")
X_imputed = X.skb.apply(imputer)

model = lgb.LGBMRegressor(
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

pred = X_imputed.skb.apply(model, y=y)

# Compile a learner from the DataOps graph
learner = pred.skb.make_learner(fitted=True)

# Validation split for score reporting
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE
)

train_split = train_df.iloc[train_idx].copy()
valid_split = train_df.iloc[valid_idx].copy()

# Fit a clean model on the training split using the same preprocessing path
split_data = skrub.var("data", train_split)
split_X = split_data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
split_y = split_data[TARGET].skb.mark_as_y()
split_X_imputed = split_X.skb.apply(SimpleImputer(strategy="median"))
split_pred = split_X_imputed.skb.apply(
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
    y=split_y,
)
split_learner = split_pred.skb.make_learner(fitted=True)

valid_env = {"data": valid_split}
valid_preds = split_learner.predict(valid_env)
final_validation_score = mean_squared_error(valid_split[TARGET].values, valid_preds) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit final learner on full training data and predict test
full_learner = learner
test_env = {"data": test_df}
test_preds = full_learner.predict(test_env)

submission = pd.DataFrame({TARGET: test_preds})
submission.to_csv("submission.csv", index=False)
