
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Minimal fix: use the already-loaded DataFrame directly for splitting.
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Keep the DataOps pipeline structure intact.
model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).reshape(-1)
valid_pred = valid_pred.astype(float)
valid_label = valid_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col].astype(bool), valid_label)
print(f"Final Validation Performance: {final_validation_score}")

# Train on full data for test prediction without changing the validation flow.
full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).reshape(-1)
test_label = test_pred.astype(float) >= 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_label.astype(bool),
    }
)
submission.to_csv("./submission.csv", index=False)
