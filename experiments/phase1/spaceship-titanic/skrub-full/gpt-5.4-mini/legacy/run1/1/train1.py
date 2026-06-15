
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
model = HistGradientBoostingClassifier(random_state=42)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)

valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).astype(bool)
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data for test prediction
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).astype(bool)

submission = pd.DataFrame(
    {"PassengerId": pd.read_csv("./input/test.csv")["PassengerId"], "Transported": test_pred}
)
submission.to_csv("submission.csv", index=False)
