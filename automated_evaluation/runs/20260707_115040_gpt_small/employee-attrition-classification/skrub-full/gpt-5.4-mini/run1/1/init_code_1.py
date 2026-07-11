
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# -----------------------------
# Config
# -----------------------------
random_state = 42
target_col = "Attrition"
id_col = "EmployeeNumber"

# -----------------------------
# Load data
# -----------------------------
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Some datasets use "id" instead of EmployeeNumber; preserve requested submission id if present.
if id_col not in train_df.columns and "id" in train_df.columns:
    id_col = "id"

# -----------------------------
# Holdout split
# -----------------------------
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# -----------------------------
# DataOps graph
# -----------------------------
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

# Smallest fix for clone-safe CatBoost in skrub:
# do NOT pass cat_features=[]; let CatBoost infer categorical handling from encoded inputs.
model = CatBoostClassifier(
    verbose=0,
    random_state=random_state,
    loss_function="Logloss",
    iterations=300,
    depth=6,
    learning_rate=0.08,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# -----------------------------
# Fit on full training data and predict test for submission
# -----------------------------
full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

full_pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred_graph.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame(
    {
        id_col: test_df[id_col].values,
        target_col: np.asarray(test_pred).ravel(),
    }
)

submission.to_csv("submission.csv", index=False)
