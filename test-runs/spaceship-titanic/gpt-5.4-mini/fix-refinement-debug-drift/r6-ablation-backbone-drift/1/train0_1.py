
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
input_dir = "./input"
train_path = os.path.join(input_dir, "train.csv")
test_path = os.path.join(input_dir, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"

# Basic feature cleanup / parsing
def preprocess_df(df):
    df = df.copy()

    # Cabin splits
    cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
    if cabin_split.shape[1] >= 3:
        df["CabinDeck"] = cabin_split[0]
        df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
        df["CabinSide"] = cabin_split[2]
    else:
        df["CabinDeck"] = pd.NA
        df["CabinNum"] = np.nan
        df["CabinSide"] = pd.NA

    # Name-derived feature
    df["NameLength"] = df["Name"].astype("string").str.len()

    # Drop original high-cardinality / parsed columns
    df = df.drop(columns=["Cabin", "Name"], errors="ignore")

    # Make booleans explicit numeric/nullable for CatBoost compatibility
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_bool_dtype(df[col]) or str(df[col].dtype) == "boolean":
            df[col] = df[col].astype("Int64")

    return df


train_df = preprocess_df(train_df)
test_df = preprocess_df(test_df)

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Ensure TableVectorizer output is fully numeric to avoid CatBoost categorical dtype error
vectorizer = skrub.TableVectorizer(
    low_cardinality="drop",
    high_cardinality="drop",
)

cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=0,
    random_seed=42,
)

pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_pred = valid_pred > 0.5 if valid_pred.dtype != bool else valid_pred

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data and predict test for submission
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(cat_model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).ravel()
test_pred = test_pred > 0.5 if test_pred.dtype != bool else test_pred

submission = pd.DataFrame(
    {
        "PassengerId": pd.read_csv(test_path)["PassengerId"],
        "Transported": test_pred.astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
