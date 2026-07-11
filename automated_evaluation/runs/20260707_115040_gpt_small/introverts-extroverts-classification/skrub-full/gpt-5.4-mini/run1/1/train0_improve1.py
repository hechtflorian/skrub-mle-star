
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
test_size = 0.2  # FIX: define holdout split size before use

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "Personality"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=300,
    depth=6,
    learning_rate=0.1,
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
