

import os
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")

target_col = "Transported"

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def add_cabin_and_missing_flags(df):
    out = df.copy()

    # Cabin structure
    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", expand=True)
        out["CabinDeck"] = cabin_parts[0].replace("", np.nan)
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2].replace("", np.nan)
    else:
        out["CabinDeck"] = np.nan
        out["CabinNum"] = np.nan
        out["CabinSide"] = np.nan

    # Missingness flags for sparse expenditure columns
    sparse_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in sparse_cols:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(np.int8)

    # Drop redundant/noisy identifiers after FE
    drop_cols = [c for c in ["PassengerId", "Cabin"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out

# DataOps binding on train_part only
data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_cabin_and_missing_flags)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

# Full post-FE frame into one TableVectorizer
vectorizer = skrub.TableVectorizer()

pred = X_train.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=500,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=4.0,
        random_seed=42,
        verbose=0,
    ),
    y=y_train,
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_pred_bool = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_bool)
print(f"Final Validation Performance: {final_validation_score}")
