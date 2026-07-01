
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Preprocess with the smallest possible fix for SpentTotal
def preprocess(df):
    df = df.copy()

    # Keep existing logic, only fix numeric conversion for multiple columns
    numeric_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["SpentTotal"] = df[numeric_cols].fillna(0).sum(axis=1)

    # Simple missing value handling / feature parsing
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        else:
            df["CabinDeck"] = np.nan
            df["CabinNum"] = np.nan
            df["CabinSide"] = np.nan
        df = df.drop(columns=["Cabin"])

    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype(str).str.len()
        df = df.drop(columns=["Name"])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(df[col].median())

    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Holdout split
train_part, valid_part = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df[target_col]
)

# DataOps graph
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
learner = pred.skb.make_learner(fitted=True)

valid_pred = learner.predict({"data": valid_part})
valid_pred = np.array(valid_pred)
if valid_pred.dtype != bool and valid_pred.dtype != np.bool_:
    valid_pred = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Train on full data and create submission
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.array(test_pred)
if test_pred.dtype != bool and test_pred.dtype != np.bool_:
    test_pred = test_pred > 0.5

submission = pd.DataFrame({
    "PassengerId": pd.read_csv("./input/test.csv")["PassengerId"],
    "Transported": test_pred.astype(bool)
})
submission.to_csv("submission.csv", index=False)
