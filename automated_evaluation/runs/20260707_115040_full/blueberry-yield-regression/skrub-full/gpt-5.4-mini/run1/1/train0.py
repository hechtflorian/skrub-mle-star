
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_df = pd.read_csv("./input/train.csv")

target_col = "yield"
random_state = 42
test_size = 0.2

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=5000,
    loss_function="MAE",
    random_seed=random_state,
    verbose=0,
)

xgb_model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=random_state,
    n_jobs=1,
    verbosity=0,
)

cat_pred_graph = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
cat_learner = cat_pred_graph.skb.make_learner(fitted=True)
cat_valid_pred = np.asarray(cat_learner.predict({"data": valid_part}), dtype=float).ravel()

xgb_pred_graph = X_train.skb.apply(vectorizer).skb.apply(xgb_model, y=y_train)
xgb_learner = xgb_pred_graph.skb.make_learner(fitted=True)
xgb_valid_pred = np.asarray(xgb_learner.predict({"data": valid_part}), dtype=float).ravel()

valid_pred = 0.5 * cat_valid_pred + 0.5 * xgb_valid_pred
final_validation_score = mean_absolute_error(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
