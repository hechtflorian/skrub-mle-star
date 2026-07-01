
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

# Fix CatBoost import error by ensuring a compatible wheel is installed.
# Smallest possible change: install/upgrade catboost before importing it.
try:
    from catboost import CatBoostClassifier
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "catboost"])
    from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import skrub

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

# Honest validation
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool:
    valid_pred = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Train on full data and predict test for submission
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred)
if test_pred.dtype != bool:
    test_pred = test_pred > 0.5

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Transported": test_pred.astype(bool)}
)
submission.to_csv("submission.csv", index=False)
