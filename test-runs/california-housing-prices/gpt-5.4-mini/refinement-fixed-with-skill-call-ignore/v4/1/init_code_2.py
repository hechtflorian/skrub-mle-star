
import os
import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

target_col = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Basic missing value handling
for df in (train_df, test_df):
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# Validation split
rng = np.random.RandomState(42)
perm = rng.permutation(len(train_df))
valid_size = max(1, int(0.2 * len(train_df)))
valid_idx = perm[:valid_size]
train_idx = perm[valid_size:]

train_part = train_df.iloc[train_idx].reset_index(drop=True)
valid_part = train_df.iloc[valid_idx].reset_index(drop=True)

# DataOps-native setup
data = skrub.var("data", train_part)

X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# Keep the pipeline DataOps-first
vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

regressor = HistGradientBoostingRegressor(random_state=42)
pred = X_vec.skb.apply(regressor, y=y)

# Fit learner
train_learner = pred.skb.make_learner(fitted=True)

# Validation prediction: valid_part already excludes target_col, so do NOT drop it again
valid_features = valid_part.drop(columns=[target_col], errors="ignore")
valid_pred = train_learner.predict({"data": valid_features})

final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Train on full training data and predict test
full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
full_y = full_data[target_col].skb.mark_as_y()

full_X_vec = full_X.skb.apply(skrub.TableVectorizer())
full_pred = full_X_vec.skb.apply(HistGradientBoostingRegressor(random_state=42), y=full_y)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
