
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_cabin_parts(df):
    out = df.copy()
    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype("string")
        cabin_split = cabin.str.split("/", n=2, expand=True)
        out["Cabin_deck"] = cabin_split[0]
        out["Cabin_room"] = cabin_split[1]
        out["Cabin_side"] = cabin_split[2]

        cabin_room_num = pd.to_numeric(out["Cabin_room"], errors="coerce")
        out["Cabin_room_num"] = cabin_room_num
        out["Cabin_deck_side"] = out["Cabin_deck"].fillna("missing").astype(str) + "_" + out["Cabin_side"].fillna("missing").astype(str)
    return out


def to_positive_score(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        pred = learner.predict({"data": df})
        pred = np.asarray(pred)
        if pred.dtype == bool:
            return pred.astype(float)
        unique_vals = set(np.unique(pred).tolist())
        if unique_vals <= {0, 1}:
            return pred.astype(float)
        return pd.Series(pred).astype(str).isin(["True", "true", "1"]).astype(float).to_numpy()


# Shared upstream block exactly as in the current script
data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_cabin_parts)

X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Leg 1: base full feature set
vectorizer_1 = skrub.TableVectorizer()
model_1 = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)
pred_chain_1 = X_train.skb.apply(vectorizer_1).skb.apply(model_1, y=y_train)
learner_1 = pred_chain_1.skb.make_learner(fitted=True)

# Leg 2: conservative clone dropping an ID-like column if present
X_train_2 = X_train.drop(columns=["PassengerId"], errors="ignore")
vectorizer_2 = skrub.TableVectorizer()
model_2 = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)
pred_chain_2 = X_train_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train)
learner_2 = pred_chain_2.skb.make_learner(fitted=True)

# Leg 3: same base chain, but use a probability-thresholded variant from the base learner
# (kept as a lightweight derived leg from the same fixed pipeline)
# We reuse learner_1 predictions and form a hard-label score path when available.
# This is safe and keeps the preprocessing/modeling flow intact.

valid_score_1 = to_positive_score(learner_1, valid_part)
valid_score_2 = to_positive_score(learner_2, valid_part)

# Leg 3 derived score path: thresholded version of base score
valid_score_3 = (valid_score_1 >= 0.5).astype(float)

# Equal-weight blend
valid_blend_score = (valid_score_1 + valid_score_2 + valid_score_3) / 3.0
valid_pred = valid_blend_score >= 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Build submission on full training data using the same safe preprocessing path.
full_data = skrub.var("data", train_df)
full_data = full_data.skb.apply_func(add_cabin_parts)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

# Refit the same legs on the full training data
full_vectorizer_1 = skrub.TableVectorizer()
full_model_1 = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)
full_pred_chain_1 = X_full.skb.apply(full_vectorizer_1).skb.apply(full_model_1, y=y_full)
full_learner_1 = full_pred_chain_1.skb.make_learner(fitted=True)

X_full_2 = X_full.drop(columns=["PassengerId"], errors="ignore")
full_vectorizer_2 = skrub.TableVectorizer()
full_model_2 = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)
full_pred_chain_2 = X_full_2.skb.apply(full_vectorizer_2).skb.apply(full_model_2, y=y_full)
full_learner_2 = full_pred_chain_2.skb.make_learner(fitted=True)

test_score_1 = to_positive_score(full_learner_1, test_df)
test_score_2 = to_positive_score(full_learner_2, test_df)
test_score_3 = (test_score_1 >= 0.5).astype(float)

test_blend_score = (test_score_1 + test_score_2 + test_score_3) / 3.0
test_pred = test_blend_score >= 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
