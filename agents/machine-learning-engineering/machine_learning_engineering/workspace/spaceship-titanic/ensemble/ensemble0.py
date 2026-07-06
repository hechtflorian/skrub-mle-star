
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
train_df = pd.read_csv(train_path)

# Feature engineering consistent with the original dataset
def preprocess_df(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string")
        cabin_parts = cabin.str.split("/", expand=True)
        df["Deck"] = cabin_parts[0]
        df["Num"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        df["Side"] = cabin_parts[2]
    else:
        df["Deck"] = np.nan
        df["Num"] = np.nan
        df["Side"] = np.nan
    return df

train_df = preprocess_df(train_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()
vectorizer_2 = skrub.TableVectorizer()
vectorizer_3 = skrub.TableVectorizer()

model_1 = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)

model_2 = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)

model_3 = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)

# Keep intact prediction chains; blend only at the output layer.
pred_1 = X_train.skb.apply(vectorizer_1).skb.apply(model_1, y=y_train)
pred_2 = X_train.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train)
pred_3 = X_train.skb.apply(vectorizer_3).skb.apply(model_3, y=y_train)

learner_1 = pred_1.skb.make_learner(fitted=True)
learner_2 = pred_2.skb.make_learner(fitted=True)
learner_3 = pred_3.skb.make_learner(fitted=True)

def get_pred_values(learner, df):
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
        if pred.dtype.kind in "OUS":
            return (pred.astype(str) == "True").astype(float)
        return pred.astype(float).ravel()

valid_p1 = get_pred_values(learner_1, valid_part)
valid_p2 = get_pred_values(learner_2, valid_part)
valid_p3 = get_pred_values(learner_3, valid_part)

# Lightweight self-ensemble: raw prob, conservative clipped variant, and hard-vote proxy.
valid_p1 = np.clip(valid_p1, 0.0, 1.0)
valid_p2 = np.clip(0.5 * valid_p2 + 0.25, 0.0, 1.0)
valid_p3 = (valid_p3 >= 0.5).astype(float)

valid_blend = 0.7 * valid_p1 + 0.2 * valid_p2 + 0.1 * valid_p3
valid_pred = (valid_blend >= 0.5)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
