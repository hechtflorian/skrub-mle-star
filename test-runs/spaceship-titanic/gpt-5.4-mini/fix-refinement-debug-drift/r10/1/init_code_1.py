
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Smallest possible fix: remove all string-valued identifier/text columns
# that CatBoost would otherwise try to interpret as numeric features.
def preprocess(df):
    df = df.copy()
    for col in ["PassengerId", "Cabin", "Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

pred = X_train.skb.apply_func(preprocess).skb.apply(
    CatBoostClassifier(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
    ),
    y=y_train,
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_pred = np.where(valid_pred >= 0.5, True, False)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply_func(preprocess).skb.apply(
    CatBoostClassifier(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
    ),
    y=y_full,
)

full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).ravel()
test_pred = np.where(test_pred >= 0.5, True, False)

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Transported": test_pred.astype(bool)}
)
submission.to_csv("submission.csv", index=False)
