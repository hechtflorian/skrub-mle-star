
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Basic feature engineering
def add_features(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
    if "Name" in df.columns:
        df["NameLength"] = df["Name"].astype(str).str.len()
        df["Surname"] = df["Name"].astype(str).str.split().str[-1]
    df["TotalSpend"] = (
        df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Fixed estimator to replace the undefined variable
estimator = CatBoostClassifier(
    loss_function="Logloss",
    iterations=300,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
)

# Build DataOps graph
model = estimator
pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

# Fit on train_part only and validate on valid_part
learner = pred.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).reshape(-1)
valid_pred = valid_pred > 0.5 if valid_pred.dtype != bool else valid_pred
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")

# Refit on full training data for submission
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()
full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).reshape(-1)
if test_pred.dtype != bool:
    test_pred = test_pred > 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred.astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
