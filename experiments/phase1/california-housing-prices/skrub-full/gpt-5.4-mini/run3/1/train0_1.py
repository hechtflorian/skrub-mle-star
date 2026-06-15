
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "lightgbm", "-q"])

import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
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

lgb_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

cat_model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.04,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

lgb_predictor = X_train.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train)
cat_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(cat_model, y=y_train)

lgb_learner = lgb_predictor.skb.make_learner(fitted=True)
cat_learner = cat_predictor.skb.make_learner(fitted=True)

valid_pred_lgb = np.asarray(lgb_learner.predict({"data": valid_part}))
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}))

valid_pred = 0.6 * valid_pred_lgb + 0.4 * valid_pred_cat
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")
