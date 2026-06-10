
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TARGET = "median_house_value"
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep a DataOps-first flow: define data, mark X/y, and use .skb.apply(...)
data = skrub.var("data", train_df)

X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

# Split once for validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# Preserve existing DataOps structure; fix the broken .skb.apply(function) call
# by using .skb.apply_func(...) for function-based feature cleaning.
def fill_missing_numeric(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
    return df

def add_ratio_features(df):
    df = df.copy()
    eps = 1e-9
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    df["population_per_household"] = df["population"] / (df["households"] + eps)
    return df

X_clean = X.skb.apply_func(fill_missing_numeric)
X_feat = X_clean.skb.apply_func(add_ratio_features)

# Keep model unchanged in spirit: use the same regressor backbone class
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=300,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)

# Use a straightforward DataOps learner path
pred = X_feat.skb.apply(model, y=y)
learner = pred.skb.make_learner(fitted=True)

# Validation performance
valid_pred = learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[TARGET], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full data and predict test
full_pred = X_feat.skb.apply(model, y=y)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
