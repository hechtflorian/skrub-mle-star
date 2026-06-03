
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

def ensure_package(pkg_name, import_name=None):
    mod = import_name or pkg_name
    try:
        __import__(mod)
        return
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

ensure_package("lightgbm")
ensure_package("skrub")

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

DATA_DIR = "./input"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

data = skrub.var("data", train_df)

# Keep subsampling if present, but only for quick iteration; final model uses full data.
preview_data = data.skb.subsample(n=min(2000, len(train_df)))

X_preview = preview_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_preview = preview_data[target_col].skb.mark_as_y()

X_full = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data[target_col].skb.mark_as_y()

# DataOps-first feature preparation
vectorizer = skrub.TableVectorizer()

# Fit a small preview pipeline to preserve DataOps structure and validate flow.
preview_pred = X_preview.skb.apply(vectorizer).skb.apply(
    LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    ),
    y=y_preview,
)

# Final model on full data
final_pred = X_full.skb.apply(vectorizer).skb.apply(
    LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    ),
    y=y_full,
)

# Compile/finalize learner and generate validation estimate via holdout split
from sklearn.model_selection import train_test_split

train_part, val_part = train_df.copy(), None
train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)

train_part = train_df.iloc[train_idx].reset_index(drop=True)
val_part = train_df.iloc[val_idx].reset_index(drop=True)

train_data = skrub.var("data", train_part)
X_train = train_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = train_data[target_col].skb.mark_as_y()

valid_data = skrub.var("data", val_part)
X_valid = valid_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_valid = valid_data[target_col].skb.mark_as_y()

holdout_pred = X_train.skb.apply(
    skrub.TableVectorizer(),
).skb.apply(
    LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    ),
    y=y_train,
)

learner = holdout_pred.skb.make_learner(fitted=True)
val_predictions = learner.predict({"data": val_part.drop(columns=[target_col], errors="ignore")})
final_validation_score = float(mean_squared_error(val_part[target_col], val_predictions) ** 0.5)

print(f"Final Validation Performance: {final_validation_score}")

# Fit on full data for test predictions
final_learner = final_pred.skb.make_learner(fitted=True)
test_predictions = final_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
