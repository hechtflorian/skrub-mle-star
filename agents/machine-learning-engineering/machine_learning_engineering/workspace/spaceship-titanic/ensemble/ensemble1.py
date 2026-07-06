
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

vectorizer = skrub.TableVectorizer()

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)

# Keep the original DataOps path intact; diversity comes from prediction-layer post-processing only.
pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)

def get_base_prob_or_pred(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        return np.asarray(learner.predict({"data": df}), dtype=float).ravel()

# Leg 1: base output
p_base_valid = get_base_prob_or_pred(val_learner, valid_part)

# Leg 2: mild temperature-style smoothing toward 0.5
p_smooth_valid = 0.8 * p_base_valid + 0.1

# Leg 3: hard vote mapped back to float
p_vote_valid = (p_base_valid >= 0.5).astype(float)

# Blend with rank-stable weighted rule
valid_blend = 0.65 * p_base_valid + 0.25 * p_smooth_valid + 0.10 * p_vote_valid

# Tiny confidence gate to avoid borderline noise
base_decision = (p_base_valid >= 0.5).astype(float)
borderline = np.abs(valid_blend - 0.5) < 0.03
valid_blend = np.where(borderline, base_decision, valid_blend)

# Dynamic threshold for potentially imbalanced validation blend
threshold = np.median(valid_blend)
valid_pred = (valid_blend >= threshold)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
