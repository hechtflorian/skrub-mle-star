
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "xgboost", "-q"])

import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=0,
)

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
xgb_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(xgb_model, y=y_train)

cat_learner = cat_pred.skb.make_learner(fitted=True)
xgb_learner = xgb_pred.skb.make_learner(fitted=True)

valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}))
valid_pred_xgb = np.asarray(xgb_learner.predict({"data": valid_part}))

valid_pred = 0.5 * valid_pred_cat + 0.5 * valid_pred_xgb
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
