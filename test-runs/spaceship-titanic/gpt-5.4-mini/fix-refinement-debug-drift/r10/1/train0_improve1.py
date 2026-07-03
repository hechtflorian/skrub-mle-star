
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

random_state = 42
target_col = "Transported"

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()


class FeaturePrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        if "Cabin" in X.columns:
            cabin_parts = X["Cabin"].astype(str).str.split("/", expand=True)
            self.has_cabin_ = True
            self.cabin_parts_len_ = cabin_parts.shape[1]
        else:
            self.has_cabin_ = False
            self.cabin_parts_len_ = 0
        return self

    def transform(self, X):
        X = X.copy()

        if "PassengerId" in X.columns:
            X["PassengerGroup"] = X["PassengerId"].astype(str).str.split("_").str[0]
            X["PassengerNumber"] = (
                X["PassengerId"].astype(str).str.split("_").str[1].fillna("0")
            )
        else:
            X["PassengerGroup"] = ""
            X["PassengerNumber"] = "0"

        if "Cabin" in X.columns:
            cabin_parts = X["Cabin"].astype(str).str.split("/", expand=True)
            X["CabinDeck"] = cabin_parts[0].fillna("Unknown")
            X["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            X["CabinSide"] = cabin_parts[2].fillna("Unknown")
        else:
            X["CabinDeck"] = "Unknown"
            X["CabinNum"] = np.nan
            X["CabinSide"] = "Unknown"

        if "Name" in X.columns:
            X["NameLength"] = X["Name"].astype(str).str.len()
            X["Surname"] = X["Name"].astype(str).str.split().str[-1].fillna("Unknown")
        else:
            X["NameLength"] = 0
            X["Surname"] = "Unknown"

        for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
            if col not in X.columns:
                X[col] = np.nan

        X["TotalSpending"] = X[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
        X["NoSpending"] = (X["TotalSpending"] == 0).astype(int)

        if "Age" in X.columns:
            X["AgeGroup"] = pd.cut(
                X["Age"],
                bins=[-np.inf, 12, 18, 30, 50, np.inf],
                labels=["child", "teen", "young_adult", "adult", "senior"],
            ).astype(str)
        else:
            X["AgeGroup"] = "unknown"

        X = X.drop(columns=["Cabin", "Name", "PassengerId"], errors="ignore")
        return X


prep = FeaturePrep()
X_train_prep = X_train.skb.apply(prep)

numeric_cols = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "CabinNum",
    "NameLength",
    "TotalSpending",
    "NoSpending",
]
categorical_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "PassengerGroup",
    "PassengerNumber",
    "CabinDeck",
    "CabinSide",
    "Surname",
    "AgeGroup",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), categorical_cols),
    ],
    remainder="drop",
)

from sklearn.preprocessing import OneHotEncoder

model_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        ), categorical_cols),
    ],
    remainder="drop",
)

clf = LogisticRegression(max_iter=2000, random_state=random_state)

pred = X_train_prep.skb.apply(model_preprocessor).skb.apply(clf, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

if isinstance(valid_pred, pd.Series):
    valid_pred_labels = valid_pred.astype(bool)
else:
    valid_pred_labels = pd.Series(valid_pred).astype(bool)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred_labels)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

X_full_prep = X_full.skb.apply(prep)
full_pred = X_full_prep.skb.apply(model_preprocessor).skb.apply(clf, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

if isinstance(test_pred, pd.Series):
    test_pred_labels = test_pred.astype(bool)
else:
    test_pred_labels = pd.Series(test_pred).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred_labels.astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
