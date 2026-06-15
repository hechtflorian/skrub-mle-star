
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Minimal cleaning / feature engineering
def preprocess(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        df = df.drop(columns=["Cabin"], errors="ignore")
    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype("string").str.len()
        df = df.drop(columns=["Name"], errors="ignore")
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Keep the original model family goal as a tree boosting classifier,
# replacing unavailable LightGBM with sklearn's histogram gradient boosting.
model1 = HistGradientBoostingClassifier(random_state=42)

pred1 = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model1, y=y_train)
val_learner1 = pred1.skb.make_learner(fitted=True)

# Second pipeline: a thin alternative with slightly different preprocessing/model settings,
# while keeping the same DataOps pattern and a very similar model family.
# We preserve the same overall pipeline style and only vary the leaf-compatible details.
model2 = HistGradientBoostingClassifier(
    random_state=7,
    learning_rate=0.05,
    max_depth=6
)

pred2 = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model2, y=y_train)
val_learner2 = pred2.skb.make_learner(fitted=True)

# Get raw predictions if possible; fall back to class outputs as pseudo-probabilities.
def get_prob_or_label(learner, df):
    out = learner.predict({"data": df})
    out = np.asarray(out)
    if out.ndim == 2 and out.shape[1] >= 2:
        p = out[:, 1]
    else:
        p = out.astype(float)
    return p

p1_valid = get_prob_or_label(val_learner1, valid_part)
p2_valid = get_prob_or_label(val_learner2, valid_part)

# Tiny weight search on shared validation split
weights = [0.5, 0.6, 0.7, 0.8]
best_w = 0.5
best_score = -1.0

for w in weights:
    p_ens = w * p1_valid + (1.0 - w) * p2_valid
    valid_pred = (p_ens >= 0.5)
    score = accuracy_score(valid_part[target_col], valid_pred)
    if score > best_score:
        best_score = score
        best_w = w

final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data for test prediction
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred1 = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model1, y=y_full)
full_learner1 = full_pred1.skb.make_learner(fitted=True)

full_pred2 = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model2, y=y_full)
full_learner2 = full_pred2.skb.make_learner(fitted=True)

def get_prob_or_label_test(learner, df):
    out = learner.predict({"data": df})
    out = np.asarray(out)
    if out.ndim == 2 and out.shape[1] >= 2:
        p = out[:, 1]
    else:
        p = out.astype(float)
    return p

p1_test = get_prob_or_label_test(full_learner1, test_df)
p2_test = get_prob_or_label_test(full_learner2, test_df)

p_ens_test = best_w * p1_test + (1.0 - best_w) * p2_test
test_pred = np.asarray(p_ens_test >= 0.5).astype(bool)

submission = pd.DataFrame(
    {"PassengerId": pd.read_csv("./input/test.csv")["PassengerId"], "Transported": test_pred}
)
submission.to_csv("submission.csv", index=False)
