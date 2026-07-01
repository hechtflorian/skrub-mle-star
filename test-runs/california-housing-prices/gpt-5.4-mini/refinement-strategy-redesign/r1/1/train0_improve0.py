
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import skrub
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "median_house_value"

# Basic cleanup
for df in (train_df, test_df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace({"": np.nan, "NA": np.nan, "NaN": np.nan})

# Split a validation set for scoring
train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=42)

data = skrub.var("data", train_part)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# DataOps-first feature pipeline
vectorizer = skrub.TableVectorizer()

regressor = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=300,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)

pred = X.skb.apply(vectorizer).skb.apply(regressor, y=y)

# Fit on training split and validate
learner = pred.skb.make_learner(fitted=True)

val_features = val_part.drop(columns=target_col, errors="ignore")
val_pred = learner.predict({"data": val_features})

final_validation_score = mean_squared_error(val_part[target_col], val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Refit on full training data and predict test set
full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
full_y = full_data[target_col].skb.mark_as_y()

full_pred = full_X.skb.apply(vectorizer).skb.apply(regressor, y=full_y)
full_learner = full_pred.skb.make_learner(fitted=True)

test_features = test_df.copy()
test_predictions = full_learner.predict({"data": test_features})

submission = pd.DataFrame({"median_house_value": test_predictions})
submission.to_csv("submission.csv", index=False)
