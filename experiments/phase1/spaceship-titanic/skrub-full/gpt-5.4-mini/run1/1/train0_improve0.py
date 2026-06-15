
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "Transported"

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps bind on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Light feature engineering
def add_features(df):
    df = df.copy()
    cabin = df["Cabin"].astype("string")
    cabin_split = cabin.str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    return df.drop(columns=["Cabin"])

# Keep backbone estimator class
model = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    StandardScaler(with_mean=False),
    LogisticRegression(max_iter=1000, random_state=42),
)

# Apply feature engineering before vectorization/model
X_train_fe = X_train.skb.apply_func(add_features)
predictor = X_train_fe.skb.apply(
    skrub.TableVectorizer(),
).skb.apply(model, y=y_train)

# Validation prediction: FIXED to use same DataOps-style env binding
val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = np.asarray(val_learner.predict({"data": valid_part})).ravel()

# Convert boolean targets if needed
y_valid = valid_part[target_col].astype(bool).to_numpy()
final_validation_score = accuracy_score(y_valid, valid_pred.astype(bool))
print(f"Final Validation Performance: {final_validation_score}")
