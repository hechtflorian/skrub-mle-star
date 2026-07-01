
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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

# Base model from the original pipeline.
hgb_model = HistGradientBoostingRegressor(random_state=42)
hgb_pred = X_vec.skb.apply(hgb_model, y=y)
hgb_learner = hgb_pred.skb.make_learner(fitted=True)

# Reference model integrated as an additional learner.
rf_model = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=2,
    random_state=0,
    n_jobs=-1,
)
rf_pred = X_vec.skb.apply(rf_model, y=y)
rf_learner = rf_pred.skb.make_learner(fitted=True)

# Validation uses features only; no target column is present in valid_part.
valid_features = valid_part.drop(columns=[TARGET], errors="ignore")
hgb_valid_preds = hgb_learner.predict({"data": valid_features})
rf_valid_preds = rf_learner.predict({"data": valid_features})

# Simple ensemble: average predictions from both models.
valid_preds = (np.asarray(hgb_valid_preds) + np.asarray(rf_valid_preds)) / 2.0
final_validation_score = mean_squared_error(valid_part[TARGET], valid_preds) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data for test predictions.
full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
full_y = full_data[TARGET].skb.mark_as_y()

full_hgb_pred = full_X.skb.apply(vectorizer).skb.apply(hgb_model, y=full_y)
full_rf_pred = full_X.skb.apply(vectorizer).skb.apply(rf_model, y=full_y)

full_hgb_learner = full_hgb_pred.skb.make_learner(fitted=True)
full_rf_learner = full_rf_pred.skb.make_learner(fitted=True)

hgb_test_preds = full_hgb_learner.predict({"data": test_df})
rf_test_preds = full_rf_learner.predict({"data": test_df})
test_preds = (np.asarray(hgb_test_preds) + np.asarray(rf_test_preds)) / 2.0

submission = pd.DataFrame({TARGET: test_preds})
submission.to_csv("submission.csv", index=False)
