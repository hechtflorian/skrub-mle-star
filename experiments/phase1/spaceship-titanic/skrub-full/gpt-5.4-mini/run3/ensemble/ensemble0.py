
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])

import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def fit_predict_proba(train_frame, predict_frame, seed=42):
    data_train = skrub.var("data", train_frame)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = lgb.LGBMClassifier(
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred.skb.make_learner(fitted=True)

    proba = learner.predict_proba({"data": predict_frame})
    proba = np.asarray(proba)

    if proba.ndim == 2:
        if hasattr(learner, "classes_") and len(getattr(learner, "classes_", [])) == 2:
            pos_idx = int(np.where(np.asarray(learner.classes_) == True)[0][0]) if True in list(learner.classes_) else 1
        else:
            pos_idx = 1
        return proba[:, pos_idx]
    return proba

# Shared split, deterministic thin ensemble copies
seeds = [42, 43, 44]

valid_probs = []
for seed in seeds:
    valid_probs.append(fit_predict_proba(train_part, valid_part, seed=seed))

blended_valid_prob = np.mean(np.vstack(valid_probs), axis=0)
valid_pred = blended_valid_prob >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
