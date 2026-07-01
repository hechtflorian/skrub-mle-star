
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Simple feature engineering inside a DataOps deferred function
@skrub.deferred
def add_features(df):
    out = df.copy()

    cabin_split = out["Cabin"].astype("string").str.split("/", expand=True)
    out["CabinDeck"] = cabin_split[0]
    out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    out["CabinSide"] = cabin_split[2]

    out["Group"] = out["PassengerId"].astype("string").str.split("_").str[0]
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["Spending"] = out[spending_cols].fillna(0).sum(axis=1)
    out["ZeroSpending"] = (out["Spending"] == 0).astype(int)
    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-np.inf, 12, 18, 30, 50, np.inf],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("string")

    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["HasName"] = out["Name"].notna().astype(int)
    return out

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps-native training graph: bind train_part only
data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)
X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    verbose=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

# Honest validation
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
if valid_pred.dtype != bool:
    valid_pred = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: refit on full training data and predict test
data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)
X_full = data_full_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).ravel()
if test_pred.dtype != bool:
    test_pred = test_pred > 0.5

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Transported": test_pred.astype(bool)}
)
submission.to_csv("submission.csv", index=False)
