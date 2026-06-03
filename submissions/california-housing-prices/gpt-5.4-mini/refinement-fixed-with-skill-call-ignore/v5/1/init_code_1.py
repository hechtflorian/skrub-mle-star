
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Install missing dependency if needed, then import skrub
try:
    import skrub
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub", "-q"])
    import skrub

# Load data
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

# DataOps-style setup
data = skrub.var("data", train_df)
X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# Split for validation
idx = np.arange(len(train_df))
train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

train_part = train_df.iloc[train_idx].copy()
val_part = train_df.iloc[val_idx].copy()

# Build DataOps pipeline
data_train = skrub.var("data_train", train_part)
X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Optional subsampling for faster iteration; kept as requested
X_train = X_train.skb.subsample(n=min(5000, len(train_part)))

vectorizer = skrub.TableVectorizer()

# Model
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=300,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)

# Keep the DataOps architecture intact
pred_plan = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

# Fit via learner compilation
learner = pred_plan.skb.make_learner(fitted=True)

# Validation
val_features = val_part.drop(columns=[target_col], errors="ignore")
val_true = val_part[target_col].to_numpy()
val_pred = learner.predict({"data_train": val_part})

final_validation_score = mean_squared_error(val_true, val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Refit on full training data for test predictions
full_data = skrub.var("full_data", train_df)
full_X = full_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
full_y = full_data[target_col].skb.mark_as_y()

full_pred_plan = full_X.skb.apply(vectorizer).skb.apply(model, y=full_y)
full_learner = full_pred_plan.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"full_data": train_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
