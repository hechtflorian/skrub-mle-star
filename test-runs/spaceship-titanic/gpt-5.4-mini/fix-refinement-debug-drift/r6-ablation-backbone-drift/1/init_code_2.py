
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph on holdout training partition only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Simple feature engineering inside the workflow
def add_features(df):
    df = df.copy()
    numeric_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TotalSpend"] = df[numeric_cols].fillna(0).sum(axis=1)
    df["IsSpender"] = (df["TotalSpend"] > 0).astype(int)
    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    if cabin_split.shape[1] >= 3:
        df["CabinDeck"] = cabin_split[0]
        df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
        df["CabinSide"] = cabin_split[2]
    else:
        df["CabinDeck"] = np.nan
        df["CabinNum"] = np.nan
        df["CabinSide"] = np.nan
    name_split = df["Name"].astype(str).str.split(" ", n=1, expand=True)
    df["NameLen"] = df["Name"].astype(str).str.len()
    df["FamilyNameLen"] = name_split[1].astype(str).str.len() if name_split.shape[1] > 1 else np.nan
    return df.drop(columns=["Cabin", "Name"], errors="ignore")

vectorizer = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.StringEncoder(),
)

model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    verbose=-1,
)

pred = (
    X_train.skb.apply_func(add_features)
    .skb.apply(vectorizer)
    .skb.apply(model, y=y_train)
)

learner = pred.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).astype(int)
valid_score = accuracy_score(valid_part[target_col].astype(int), valid_pred)

final_validation_score = valid_score
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: train on full training data and predict test
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = (
    X_full.skb.apply_func(add_features)
    .skb.apply(vectorizer)
    .skb.apply(model, y=y_full)
)

full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
