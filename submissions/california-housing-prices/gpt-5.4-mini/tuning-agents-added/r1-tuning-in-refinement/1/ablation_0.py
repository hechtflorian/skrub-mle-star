
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import skrub

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Basic cleaning
for df in (train_df, test_df):
    for col in df.columns:
        if df[col].dtype.kind in "if":
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# Fill missing values with train medians
feature_cols = [c for c in train_df.columns if c != TARGET_COL]
medians = train_df[feature_cols].median(numeric_only=True)
train_df[feature_cols] = train_df[feature_cols].fillna(medians)
test_df[feature_cols] = test_df[feature_cols].fillna(medians)

# DataOps graph
data = skrub.var("data", train_df)
X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y = data[TARGET_COL].skb.mark_as_y()

def add_feature_engineering(df):
    df = df.copy()
    eps = 1e-6
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    df["population_per_household"] = df["population"] / (df["households"] + eps)
    df["rooms_per_person"] = df["total_rooms"] / (df["population"] + eps)
    df["bedrooms_per_household"] = df["total_bedrooms"] / (df["households"] + eps)
    return df

# Fix: use apply_func for plain Python function
X_feat = X.skb.apply_func(add_feature_engineering)

# Keep the same model family; use a strong fixed baseline
model = HistGradientBoostingRegressor(
    learning_rate=0.08,
    max_depth=8,
    max_iter=300,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=RANDOM_STATE,
)

pred = X_feat.skb.apply(model, y=y)
learner = pred.skb.make_learner(fitted=True)

# Validation split for performance reporting
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train_part = train_df.iloc[train_idx].reset_index(drop=True)
val_part = train_df.iloc[val_idx].reset_index(drop=True)

# Rebuild the same DataOps path for validation
data_val = skrub.var("data", train_part)
X_val = data_val.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_val = data_val[TARGET_COL].skb.mark_as_y()
X_val_feat = X_val.skb.apply_func(add_feature_engineering)

pred_val = X_val_feat.skb.apply(model, y=y_val)
learner_val = pred_val.skb.make_learner(fitted=True)

val_env = {"data": val_part}
val_pred = learner_val.predict(val_env)
final_validation_score = mean_squared_error(val_part[TARGET_COL], val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data and predict test set
full_learner = learner
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head().to_string(index=False))
