
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reproducibility
random_state = 42
np.random.seed(random_state)

# Load data
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"

# Basic preprocessing helpers
def preprocess_df(df):
    df = df.copy()

    # Split Cabin into useful parts
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string")
        cabin_split = cabin.str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        else:
            df["CabinDeck"] = pd.NA
            df["CabinNum"] = np.nan
            df["CabinSide"] = pd.NA
        df = df.drop(columns=["Cabin"], errors="ignore")

    # Family/group size proxy from name
    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype("string").str.split().str[-1]
        df = df.drop(columns=["Name"], errors="ignore")

    # Fill common missing values
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0)

    return df

train_df = preprocess_df(train_df)
test_df = preprocess_df(test_df)

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps pipeline
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
predictor = X_train.skb.apply(vectorizer).skb.apply(
    LGBMClassifier(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    ),
    y=y_train,
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred_label = (valid_pred > 0.5).astype(bool) if valid_pred.dtype != bool else valid_pred
final_validation_score = accuracy_score(valid_part[target_col], valid_pred_label)

print(f"Final Validation Performance: {final_validation_score}")

# Train on full data and predict test for submission
full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

full_predictor = X_full.skb.apply(vectorizer).skb.apply(
    LGBMClassifier(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    ),
    y=y_full,
)
full_learner = full_predictor.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred_label = (test_pred > 0.5).astype(bool) if test_pred.dtype != bool else test_pred

submission = pd.DataFrame(
    {
        "PassengerId": pd.read_csv(test_path)["PassengerId"],
        "Transported": test_pred_label,
    }
)
submission.to_csv("submission.csv", index=False)
