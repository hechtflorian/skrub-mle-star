
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "NObeyesdad"
random_state = 42

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps pipeline on holdout
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=400,
    depth=6,
    learning_rate=0.08,
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)

valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).reshape(-1)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Full-train refit for submission
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).reshape(-1)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"id": test_df["id"], "NObeyesdad": test_pred})
submission.to_csv("./final/submission.csv", index=False)
