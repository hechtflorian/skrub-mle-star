
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"
random_state = 42

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

predictor = X_train.skb.apply(
    skrub.TableVectorizer()
).skb.apply(
    CatBoostClassifier(
        verbose=0,
        random_state=random_state,
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
    ),
    y=y_train,
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool:
    valid_pred = valid_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_predictor = X_full.skb.apply(
    skrub.TableVectorizer()
).skb.apply(
    CatBoostClassifier(
        verbose=0,
        random_state=random_state,
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
    ),
    y=y_full,
)

full_learner = full_predictor.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred)
if test_pred.dtype != bool:
    test_pred = test_pred >= 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred.astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
