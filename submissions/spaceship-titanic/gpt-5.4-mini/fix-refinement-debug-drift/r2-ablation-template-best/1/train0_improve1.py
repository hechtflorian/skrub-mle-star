
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "Transported"
metric_fn = accuracy_score

# Minimal helper used by existing ablation scoring section
def score_variant(variant_name, build_variant_name, train_part, valid_part, target_col, metric_fn):
    # Keep this placeholder-compatible with the original workflow:
    # the reported bug was an undefined metric_fn, so we only need to ensure it exists.
    raise NotImplementedError("This script should not reach this helper in the fixed path.")

# Split for honest holdout validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps pipeline
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Keep backbone estimator class the same family as a standard sklearn classifier
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).astype(bool)

final_validation_score = metric_fn(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
