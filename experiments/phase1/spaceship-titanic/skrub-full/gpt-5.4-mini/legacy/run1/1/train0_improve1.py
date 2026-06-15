
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "Transported"

# Minimal fix: define the missing preprocess function
def preprocess(df):
    df = df.copy()

    # Parse Cabin into useful components
    cabin = df["Cabin"].fillna("Unknown/Unknown/Unknown").astype(str).str.split("/", expand=True)
    if cabin.shape[1] >= 3:
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
    else:
        df["CabinDeck"] = np.nan
        df["CabinNum"] = np.nan
        df["CabinSide"] = np.nan

    # Family/group features from PassengerId
    if "PassengerId" in df.columns:
        df["GroupId"] = df["PassengerId"].astype(str).str.split("_").str[0]
        df["GroupSize"] = df.groupby("GroupId")["PassengerId"].transform("count")
    else:
        df["GroupId"] = np.nan
        df["GroupSize"] = np.nan

    # Total spend
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        if col not in df.columns:
            df[col] = np.nan
    df["TotalSpend"] = df[spend_cols].fillna(0).sum(axis=1)

    # Fill numeric missing values
    numeric_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum", "GroupSize", "TotalSpend"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    cat_cols = [
        "HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck",
        "CabinSide", "GroupId", "Name"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("Missing")

    return df

# Preserve existing call pattern
train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps pipeline
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
# Keep same backbone estimator class pattern; use a simple sklearn model if already present logic is missing.
# Since the original code referenced preprocess only, we keep the downstream logic minimal and stable.
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
