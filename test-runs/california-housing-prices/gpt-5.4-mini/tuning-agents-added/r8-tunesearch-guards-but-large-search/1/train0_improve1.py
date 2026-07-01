
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

TARGET_COL = "median_house_value"
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep the same split contract for honest holdout validation.
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps block: bind only train_part for validation scoring.
data_train = skrub.var("data", train_part)

# Simple feature cleanup while preserving DataOps structure.
# The original bug was using DropCols via ".skb.apply"; DropCols is a transformer
# and should be passed directly to .skb.apply(...).
X_train_fe = data_train.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_train = data_train[TARGET_COL].skb.mark_as_y()

# Correct API usage: no ".skb" on DropCols itself.
X_train_fe = X_train_fe.skb.apply(skrub.DropCols(cols=["households"]))

# Keep the backbone estimator class as a tree-based regressor.
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=250,
    random_state=42,
    l2_regularization=0.0,
)

# Vectorize/encode the post-FE table, then fit the model in the DataOps graph.
pred = X_train_fe.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = mean_squared_error(valid_part[TARGET_COL], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Submission-stage only: refit on full train_df and predict test_df.
# This is placed after validation print to preserve honest holdout binding.
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_full = data_full[TARGET_COL].skb.mark_as_y()

X_full = X_full.skb.apply(skrub.DropCols(cols=["households"]))
full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)
