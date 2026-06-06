
import os
import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

if len(train_df) > 20000:
    train_df = train_df.sample(n=20000, random_state=42).reset_index(drop=True)

rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_split = train_df.iloc[tr_idx].reset_index(drop=True)
valid_split = train_df.iloc[va_idx].reset_index(drop=True)

data = skrub.var("data", train_split)
X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y = data[TARGET_COL].skb.mark_as_y()

vectorizer1 = skrub.TableVectorizer()
model1 = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pred_graph1 = X.skb.apply(vectorizer1).skb.apply(model1, y=y)
learner1 = pred_graph1.skb.make_learner(fitted=True)
learner1.fit({"data": train_split})
valid_pred1 = learner1.predict({"data": valid_split})


