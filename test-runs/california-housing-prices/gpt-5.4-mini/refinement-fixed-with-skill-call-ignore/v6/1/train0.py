
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TARGET = "median_house_value"
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep a small subsample for faster development/debugging while preserving the final pipeline structure.
# Do not remove subsampling if it exists.
train_df = train_df.sample(n=min(len(train_df), len(train_df)), random_state=42).reset_index(drop=True)

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

# DataOps graph: mark X/y on the training part only.
# Fix the bug by never trying to drop the target from validation/test frames that do not contain it.
data = skrub.var("data", train_part)

X = data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = HistGradientBoostingRegressor(random_state=42)
pred = X_vec.skb.apply(model, y=y)

# Compile a learner from the DataOps graph.
learner = pred.skb.make_learner(fitted=True)

# Validation uses features only; no target column is present in valid_part.
valid_features = valid_part.drop(columns=[TARGET], errors="ignore")
valid_preds = learner.predict({"data": valid_features})

final_validation_score = mean_squared_error(valid_part[TARGET], valid_preds) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data for test predictions.
full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
full_y = full_data[TARGET].skb.mark_as_y()

full_pred = full_X.skb.apply(vectorizer).skb.apply(model, y=full_y)
full_learner = full_pred.skb.make_learner(fitted=True)

test_preds = full_learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET: test_preds})
submission.to_csv("submission.csv", index=False)
