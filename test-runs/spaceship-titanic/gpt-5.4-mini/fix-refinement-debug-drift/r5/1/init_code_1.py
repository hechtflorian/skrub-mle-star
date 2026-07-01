
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"

# Simple, safe preprocessing to ensure CatBoost receives only numeric features.
# Keep the DataOps pipeline structure intact while avoiding raw string columns.
def preprocess_df(df):
    df = df.copy()
    if "PassengerId" in df.columns:
        df["PassengerGroup"] = df["PassengerId"].astype(str).str.split("_").str[0]
        df["PassengerNum"] = df["PassengerId"].astype(str).str.split("_").str[1]
        df = df.drop(columns=["PassengerId"], errors="ignore")

    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
        df["CabinDeck"] = cabin_split[0]
        df["CabinNum"] = cabin_split[1]
        df["CabinSide"] = cabin_split[2]
        df = df.drop(columns=["Cabin"], errors="ignore")

    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype(str).str.len()
        df = df.drop(columns=["Name"], errors="ignore")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

    return df

train_df_proc = preprocess_df(train_df)
test_df_proc = preprocess_df(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df_proc)), test_size=0.2, random_state=42
)
train_part = train_df_proc.iloc[train_idx].copy()
valid_part = train_df_proc.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
    iterations=300,
    depth=6,
    learning_rate=0.05,
)

pred_graph = X_train.skb.apply(model, y=y_train)
val_learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred_binary = np.asarray(valid_pred) > 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred_binary)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: fit on the full training data and predict on test data.
data_full = skrub.var("data", train_df_proc)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()
full_pred_graph = X_full.skb.apply(model, y=y_full)
full_learner = full_pred_graph.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df_proc})
test_pred_binary = np.asarray(test_pred) > 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred_binary.astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
