
import os
import json
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

# Terminal tuning: inject choose_* only on the model block
model = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.03, 0.08, log=False, default=0.05, name="learning_rate"),
    max_depth=skrub.choose_int(6, 10, default=8, name="max_depth"),
    max_iter=300,
    min_samples_leaf=skrub.choose_int(10, 30, default=20, name="min_samples_leaf"),
    l2_regularization=0.0,
    random_state=42,
)

# Use a straightforward DataOps learner path
pred = X_feat.skb.apply(model, y=y)

# Real holdout randomized search from the final prediction DataOp
search = pred.skb.make_randomized_search(n_iter=4, n_jobs=2, random_state=42, fitted=True)
search.fit({"data": train_part})

# Validation performance
valid_pred = search.best_learner_.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[TARGET], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

best_params = {}
if hasattr(search, "best_params_") and search.best_params_ is not None:
    for k, v in search.best_params_.items():
        if hasattr(v, "item"):
            v = v.item()
        best_params[k] = v
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))

# Fit on full data and predict test
full_pred = X_feat.skb.apply(model, y=y)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
