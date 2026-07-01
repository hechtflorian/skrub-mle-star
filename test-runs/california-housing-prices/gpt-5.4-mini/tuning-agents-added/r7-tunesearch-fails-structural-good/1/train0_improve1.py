
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "median_house_value"

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps-native bindings
data_train = skrub.var("data", train_part)

# Fix: avoid eager membership test on X_train.columns
# Use Skrub-compatible drop with errors="ignore"
X_train = data_train.drop(columns=[target_col, "households"], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Keep the same backbone estimator class
model = Ridge(alpha=1.0, random_state=42)

# DataOps pipeline
pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

# Honest validation: fit on train_part only, predict valid_part
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: fit on full train and predict test
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=[target_col, "households"], errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
