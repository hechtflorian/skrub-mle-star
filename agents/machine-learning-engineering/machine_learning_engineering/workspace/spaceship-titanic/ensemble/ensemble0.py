
import os
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "Transported"

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Minimal fix: vectorize categorical/string columns before CatBoost
# so CatBoost receives only numeric features.
vectorizer = skrub.TableVectorizer()

pred1 = X_train.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_train,
)

pred2 = X_train.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=2024,
        verbose=0,
    ),
    y=y_train,
)

val_learner1 = pred1.skb.make_learner(fitted=True)
val_learner2 = pred2.skb.make_learner(fitted=True)

valid_pred1 = np.asarray(val_learner1.predict({"data": valid_part})).ravel()
valid_pred2 = np.asarray(val_learner2.predict({"data": valid_part})).ravel()
valid_pred = 0.5 * valid_pred1 + 0.5 * valid_pred2
valid_pred_bool = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_bool)
print(f"Final Validation Performance: {final_validation_score}")

# Submission-stage refit on full training data
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred1 = X_full.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_full,
)

full_pred2 = X_full.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=2024,
        verbose=0,
    ),
    y=y_full,
)

full_learner1 = full_pred1.skb.make_learner(fitted=True)
full_learner2 = full_pred2.skb.make_learner(fitted=True)

test_pred1 = np.asarray(full_learner1.predict({"data": test_df})).ravel()
test_pred2 = np.asarray(full_learner2.predict({"data": test_df})).ravel()
test_pred = 0.5 * test_pred1 + 0.5 * test_pred2
test_pred_bool = test_pred > 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred_bool.astype(bool),
    }
)
submission["Transported"] = submission["Transported"].map({True: "True", False: "False"})
submission.to_csv("submission.csv", index=False)
