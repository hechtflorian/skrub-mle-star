
import os
import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Hold-out validation split
rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_split = train_df.iloc[tr_idx].reset_index(drop=True)
valid_split = train_df.iloc[va_idx].reset_index(drop=True)

# DataOps graph
data = skrub.var("data", train_split)
X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y = data[TARGET_COL].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

regressor = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pred_graph = X.skb.apply(vectorizer).skb.apply(regressor, y=y)
learner = pred_graph.skb.make_learner(fitted=True)
learner.fit({"data": train_split})

valid_pred = learner.predict({"data": valid_split})
final_validation_score = mean_squared_error(valid_split[TARGET_COL], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full data and predict test
full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y_full = full_data[TARGET_COL].skb.mark_as_y()

full_pred_graph = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(
    LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
    y=y_full,
)
full_learner = full_pred_graph.skb.make_learner(fitted=True)
full_learner.fit({"data": train_df})

test_pred = full_learner.predict({"data": test_df})
submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)
