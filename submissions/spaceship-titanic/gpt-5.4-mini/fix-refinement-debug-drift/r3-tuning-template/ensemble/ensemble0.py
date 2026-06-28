
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Split for honest holdout validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding on train_part only for validation
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Lightweight feature engineering within DataOps path
def fe(df):
    df = df.copy()
    cabin = df["Cabin"].astype("string").str.split("/", expand=True)
    if cabin.shape[1] >= 3:
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype("string").str.split().str[-1]
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing = [c for c in spend_cols if c in df.columns]
    df["TotalSpend"] = df[existing].fillna(0).sum(axis=1) if existing else 0
    if existing:
        df["NoSpend"] = (df[existing].fillna(0).sum(axis=1) == 0)
    return df

X_train = X_train.skb.apply_func(fe)

vectorizer = skrub.TableVectorizer()
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

# Ensure boolean labels for accuracy
valid_true = valid_part[target_col].astype(bool).to_numpy()
valid_pred = np.asarray(valid_pred).astype(bool)
final_validation_score = accuracy_score(valid_true, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission-stage refit on full training data
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()
X_full = X_full.skb.apply_func(fe)

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": np.asarray(test_pred).astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
