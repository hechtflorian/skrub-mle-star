
import os
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
input_dir = "./input"
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

target_col = "Transported"
random_state = 42

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps bind on train_part only (avoid leakage)
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Minimal fix: exclude raw string identifier/name columns so CatBoost never sees PassengerId as numeric
def drop_raw_string_ids(df):
    return df.drop(columns=["PassengerId", "Name"], errors="ignore")

model = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

pred = (
    X_train.skb.apply_func(drop_raw_string_ids)
    .skb.apply(skrub.TableVectorizer())
    .skb.apply(model, y=y_train)
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_pred_labels = valid_pred > 0.5 if valid_pred.dtype != bool else valid_pred

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_labels)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = (
    X_full.skb.apply_func(drop_raw_string_ids)
    .skb.apply(skrub.TableVectorizer())
    .skb.apply(model, y=y_full)
)

full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).ravel()
test_pred_labels = test_pred > 0.5 if test_pred.dtype != bool else test_pred

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred_labels.astype(bool),
    }
)
submission["Transported"] = submission["Transported"].map({True: "True", False: "False"})
submission.to_csv("submission.csv", index=False)
