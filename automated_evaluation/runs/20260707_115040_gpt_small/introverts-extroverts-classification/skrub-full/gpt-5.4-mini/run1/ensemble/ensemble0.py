
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Personality"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_missing_indicators(df):
    out = df.copy()
    null_cols = out.columns[out.isna().any()]
    for col in null_cols:
        out[f"{col}__is_null"] = out[col].isna().astype("int8")
    return out


# Shared DataOps root
data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_missing_indicators)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Leg 1: original stabilized pipeline with missing-indicator helper
cleaner_1 = skrub.Cleaner(drop_if_constant=True)
vectorizer_1 = skrub.TableVectorizer()
model_1 = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state,
)

pred_1 = (
    X_train.skb.apply(cleaner_1)
    .skb.apply(vectorizer_1)
    .skb.apply(model_1, y=y_train)
)
learner_1 = pred_1.skb.make_learner(fitted=True)

# Leg 2: minimally different view — same backbone but without extra missing-indicator augmentation
data_train_2 = skrub.var("data", train_part)
X_train_2 = data_train_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_2 = data_train_2[target_col].skb.mark_as_y()

cleaner_2 = skrub.Cleaner(drop_if_constant=True)
vectorizer_2 = skrub.TableVectorizer()
model_2 = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state + 1,
)

pred_2 = (
    X_train_2.skb.apply(cleaner_2)
    .skb.apply(vectorizer_2)
    .skb.apply(model_2, y=y_train_2)
)
learner_2 = pred_2.skb.make_learner(fitted=True)


def _get_pred(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].ravel()
        return proba.ravel()
    except Exception:
        pred = learner.predict({"data": df})
        pred = np.asarray(pred)
        if pred.dtype.kind in "fc":
            return pred.ravel()
        return pred.astype(float).ravel()


valid_pred_1 = _get_pred(learner_1, valid_part)
valid_pred_2 = _get_pred(learner_2, valid_part)

# Soft blend if probability-like outputs are available, otherwise majority vote
blend_scores = 0.5 * valid_pred_1 + 0.5 * valid_pred_2
valid_pred = np.where(blend_scores >= 0.5, "Extrovert", "Introvert")

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
