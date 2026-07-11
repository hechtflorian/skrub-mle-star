
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


random_state = 42
target_col = "Personality"

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

# Smallest fix: remove cat_features so skrub/sklearn cloning works in make_learner(fitted=True).
# CatBoost can still consume the vectorized numeric features produced by TableVectorizer.
predictor = X_train.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    ),
    y=y_train,
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
