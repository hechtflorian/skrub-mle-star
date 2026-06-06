
import os
import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep subsampling for fast iteration if the dataset is large, but do not
# remove it entirely per instructions.
if len(train_df) > 20000:
    train_df = train_df.sample(n=20000, random_state=42).reset_index(drop=True)

# Robustly handle the presence/absence of the target column.
# Use errors="ignore" so prediction-context data without the target does not fail.
data = skrub.var("data", train_df)

X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
if TARGET_COL in train_df.columns:
    y = data[TARGET_COL].skb.mark_as_y()
else:
    # Fallback, should not happen for training data
    y = skrub.y(train_df[TARGET_COL])

# Simple, strong regression baseline with DataOps graph preserved.
vectorizer = skrub.TableVectorizer()
regressor = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=300,
    random_state=42,
)

pred_graph = X.skb.apply(vectorizer).skb.apply(regressor, y=y)

# Compile/finalize learner and fit using DataOps environment dict.
learner = pred_graph.skb.make_learner(fitted=True)

# Validation on training data via holdout split for a measurable score
rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(len(idx) * 0.8)
tr_idx, va_idx = idx[:split], idx[split:]

train_split = train_df.iloc[tr_idx].reset_index(drop=True)
valid_split = train_df.iloc[va_idx].reset_index(drop=True)

data_tr = skrub.var("data", train_split)
X_tr = data_tr.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y_tr = data_tr[TARGET_COL].skb.mark_as_y()

pred_graph_tr = X_tr.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    ),
    y=y_tr,
)
learner_tr = pred_graph_tr.skb.make_learner(fitted=True)
learner_tr.fit({"data": train_split})

valid_pred = learner_tr.predict({"data": valid_split})
final_validation_score = mean_squared_error(valid_split[TARGET_COL], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data and predict test set
learner.fit({"data": train_df})
test_pred = learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)
