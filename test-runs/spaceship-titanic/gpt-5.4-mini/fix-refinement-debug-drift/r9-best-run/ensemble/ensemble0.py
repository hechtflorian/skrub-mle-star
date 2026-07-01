
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def add_features(df):
    df = df.copy()
    cabin = df["Cabin"].fillna("X").astype(str).str.split("/")
    df["CabinDeck"] = cabin.str[0]
    df["CabinNum"] = pd.to_numeric(cabin.str[1], errors="coerce")
    df["CabinSide"] = cabin.str[2]
    df["TotalSpend"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["GroupSize"] = df["PassengerId"].astype(str).str.split("_").str[0].map(df["PassengerId"].astype(str).str.split("_").str[0].value_counts())
    df["NameLen"] = df["Name"].fillna("").astype(str).str.len()
    df["Surname"] = df["Name"].fillna("Unknown").astype(str).str.split().str[-1]
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

features = [
    "HomePlanet", "CryoSleep", "CabinDeck", "CabinSide", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "CabinNum", "TotalSpend", "NoSpend", "GroupSize", "NameLen", "Surname"
]

# Base learner 1: keep original pipeline structure exactly as-is
vectorizer1 = skrub.TableVectorizer(high_cardinality="drop")
model1 = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=24,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

pred_graph1 = (
    X_train[features]
    .skb.apply(vectorizer1)
    .skb.apply(VarianceThreshold())
    .skb.apply(model1, y=y_train)
)

# Base learner 2: a minimally different skrub pipeline, kept in the same DataOps style
# This remains a separate base learner and only the post-prediction merge logic is new.
vectorizer2 = skrub.TableVectorizer(high_cardinality="drop")
model2 = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.75,
    random_state=random_state + 7,
    verbose=-1,
)

pred_graph2 = (
    X_train[features]
    .skb.apply(vectorizer2)
    .skb.apply(VarianceThreshold())
    .skb.apply(model2, y=y_train)
)

# Fit both learners on the same holdout split
val_learner1 = pred_graph1.skb.make_learner(fitted=True)
val_learner2 = pred_graph2.skb.make_learner(fitted=True)

valid_pred1 = val_learner1.predict({"data": valid_part})
valid_pred2 = val_learner2.predict({"data": valid_part})

valid_pred1 = np.asarray(valid_pred1).reshape(-1)
valid_pred2 = np.asarray(valid_pred2).reshape(-1)

# Map to boolean / 0-1 if needed
if valid_pred1.dtype != bool:
    valid_pred1 = pd.Series(valid_pred1).astype(bool).to_numpy()
if valid_pred2.dtype != bool:
    valid_pred2 = pd.Series(valid_pred2).astype(bool).to_numpy()

acc1 = accuracy_score(valid_part[target_col], valid_pred1)
acc2 = accuracy_score(valid_part[target_col], valid_pred2)

# Weight proportional to validation accuracy
denom = acc1 + acc2
if denom <= 0:
    w1, w2 = 0.5, 0.5
else:
    w1 = acc1 / denom
    w2 = acc2 / denom

# Weighted vote with disagreement fallback
ensemble_score = w1 * valid_pred1.astype(float) + w2 * valid_pred2.astype(float)
valid_ensemble_pred = (ensemble_score >= 0.5)

# If the two predictions differ, choose the prediction from the model with higher validation accuracy
disagree = valid_pred1 != valid_pred2
if acc1 > acc2:
    valid_ensemble_pred[disagree] = valid_pred1[disagree]
elif acc2 > acc1:
    valid_ensemble_pred[disagree] = valid_pred2[disagree]
# if equal, keep weighted vote result

final_validation_score = accuracy_score(valid_part[target_col], valid_ensemble_pred)
print(f"Final Validation Performance: {final_validation_score}")
