
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier
import skrub

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

# Basic preprocessing helpers
def fe_func(df):
    out = df.copy()

    # Cabin-derived features
    cabin = out["Cabin"].astype("string")
    cabin_parts = cabin.str.split("/", expand=True)
    out["CabinDeck"] = cabin_parts[0]
    out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    out["CabinSide"] = cabin_parts[2]

    # Name-derived feature
    out["Surname"] = out["Name"].astype("string").str.split().str[-1]

    # Spend features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["TotalSpend"] = out[spend_cols].fillna(0).sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)

    # Group features from PassengerId
    out["GroupId"] = out["PassengerId"].astype("string").str.split("_").str[0]
    out["GroupSize"] = out.groupby("GroupId")["PassengerId"].transform("size")

    return out

# Binary wrapper to avoid category/string issues in CatBoost target handling
class CatBoostBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=42,
        verbose=0,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y):
        y_arr = np.asarray(y)
        if y_arr.dtype == bool:
            y_arr = y_arr.astype(int)
        elif y_arr.dtype.kind in "OUS":
            y_arr = pd.Series(y_arr).map({False: 0, True: 1}).astype(int).to_numpy()

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )
        self.model_.fit(X, y_arr)
        return self

    def predict(self, X):
        pred = self.model_.predict(X)
        pred = np.asarray(pred).reshape(-1)
        return pred.astype(bool)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

X_train_fe = X_train.skb.apply_func(fe_func)
vectorizer = skrub.TableVectorizer()
model = CatBoostBinaryClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=random_state,
    verbose=0,
)

pred = X_train_fe.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
