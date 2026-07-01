
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub", "-q"])
    import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep a small subsample for faster development if available, but do not remove it.
# Use it only when the dataset is large enough.
if len(train_df) > 5000:
    train_df = train_df.sample(n=5000, random_state=RANDOM_STATE).reset_index(drop=True)

# DataOps-first pipeline
data = skrub.var("data", train_df)
X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y = data[TARGET_COL].skb.mark_as_y()

# Simple, robust model; preserve DataOps chain structure
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=300,
    min_samples_leaf=20,
    random_state=RANDOM_STATE,
)

pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y)

# Create a train/validation split from the DataOps plan
split = pred.skb.train_test_split(random_state=RANDOM_STATE)
learner = pred.skb.make_learner()

# Fit using the environment dict, not raw X/y arrays
learner.fit(split["train"])

# Predict on the validation environment
valid_pred = learner.predict(split["test"])

# Retrieve validation target from the split environment if available
# Fallback to y_test for robustness if the split provides it separately
if "y_test" in split:
    y_valid = split["y_test"]
else:
    # In case the split only returns test env, evaluate target from original data
    # aligned through the split's test environment by re-evaluating the y DataOp.
    y_valid = pd.Series(split["test"][TARGET_COL], name=TARGET_COL)

# Ensure arrays for RMSE
y_valid_arr = np.asarray(y_valid).ravel()
valid_pred_arr = np.asarray(valid_pred).ravel()
final_validation_score = mean_squared_error(y_valid_arr, valid_pred_arr) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data and predict on test
learner_full = pred.skb.make_learner()
learner_full.fit({"data": train_df})
test_pred = learner_full.predict({"data": test_df})

submission = pd.DataFrame({TARGET_COL: np.asarray(test_pred).ravel()})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_csv(index=False).strip())
