
import os
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)

# Minimal fix: ensure non-numeric columns are routed as categorical by letting TableVectorizer
# handle them before CatBoost, and explicitly drop the target from features.
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)

pred_chain = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred_chain.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool:
    # CatBoost may emit 0/1 or class labels; normalize to boolean predictions.
    unique_vals = set(np.unique(valid_pred).tolist())
    if unique_vals <= {0, 1}:
        valid_pred = valid_pred.astype(int).astype(bool)
    else:
        valid_pred = pd.Series(valid_pred).astype(str).isin(["True", "true", "1"]).to_numpy()

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Build submission on full training data using the same safe preprocessing path.
full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

full_pred_chain = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred_chain.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

test_pred = np.asarray(test_pred)
if test_pred.dtype != bool:
    unique_vals = set(np.unique(test_pred).tolist())
    if unique_vals <= {0, 1}:
        test_pred = test_pred.astype(int).astype(bool)
    else:
        test_pred = pd.Series(test_pred).astype(str).isin(["True", "true", "1"]).to_numpy()

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
