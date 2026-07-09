
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Personality"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Minimal fix: remove cat_features so CatBoost does not try to match
# original categorical column names against transformed vectorized features.
vectorizer = skrub.TableVectorizer()

model = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data for test prediction
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"id": test_df["id"], "Personality": test_pred})
submission.to_csv("submission.csv", index=False)
