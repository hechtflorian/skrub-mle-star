
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import skrub

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

def prep(df):
    df = df.copy()

    # Keep the useful cabin/name-derived features, but avoid unstable manual category codes
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
        df["CabinDeck"] = cabin_split[0]
        df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
        df["CabinSide"] = cabin_split[2]
    else:
        df["CabinDeck"] = np.nan
        df["CabinNum"] = np.nan
        df["CabinSide"] = np.nan

    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype(str).str.split().str[-1]
    else:
        df["Surname"] = np.nan

    # Clean target-like booleans if present
    for col in ["CryoSleep", "VIP"]:
        if col in df.columns:
            df[col] = df[col].map({True: True, False: False, "True": True, "False": False})

    # Drop raw high-cardinality columns after feature extraction
    df = df.drop(columns=["Cabin", "Name"], errors="ignore")
    return df

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

train_part = prep(train_part)
valid_part = prep(valid_part)
test_prepped = prep(test_df)

# DataOps binding
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Cleaner, more stable vectorization path
vectorizer = skrub.TableVectorizer()

pred = X_train.skb.apply(vectorizer).skb.apply(
    RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    ),
    y=y_train,
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
