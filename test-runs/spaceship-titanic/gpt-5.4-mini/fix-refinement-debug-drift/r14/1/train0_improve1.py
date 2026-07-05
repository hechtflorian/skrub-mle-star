
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv(os.path.join("./input", "train.csv"))

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Fix: let TableVectorizer create the transformed feature space, then fit CatBoost
# on the vectorized output without passing stale original column-name cat_features.
pred_chain = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        random_seed=random_state,
        verbose=0,
    ),
    y=y_train,
)

val_learner = pred_chain.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()

# Ensure boolean labels for accuracy_score
if valid_pred.dtype != bool:
    valid_pred = pd.Series(valid_pred).map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).to_numpy()

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
