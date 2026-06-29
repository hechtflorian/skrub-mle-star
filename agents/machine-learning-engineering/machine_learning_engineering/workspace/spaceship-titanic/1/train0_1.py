
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "Transported"

# Basic feature engineering
def add_features(df):
    df = df.copy()
    cabin = df["Cabin"].astype("string")
    cabin_parts = cabin.str.split("/", expand=True)
    df["CabinDeck"] = cabin_parts[0]
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["CabinSide"] = cabin_parts[2]

    name = df["Name"].astype("string")
    df["Surname"] = name.str.split(" ", n=1).str[-1]
    df["NameLen"] = name.str.len()

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["SpendMean"] = df[spend_cols].fillna(0).mean(axis=1)
    df["IsAlone"] = df["PassengerId"].astype("string").str.split("_", expand=True)[1].astype(int) == 1

    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 65, 100],
        labels=["child", "teen", "young", "adult", "mid", "senior", "elder"],
    )
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=600,
    learning_rate=0.04,
    depth=6,
    random_seed=42,
    verbose=0,
)

lgb_model = LGBMClassifier(
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
lgb_pred = X_train.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train)

cat_learner = cat_pred.skb.make_learner(fitted=True)
lgb_learner = lgb_pred.skb.make_learner(fitted=True)

valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part})).ravel()
valid_pred_lgb = np.asarray(lgb_learner.predict({"data": valid_part})).ravel()

valid_pred_avg = (valid_pred_cat + valid_pred_lgb) / 2.0
valid_pred_bool = valid_pred_avg > 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_bool)
print(f"Final Validation Performance: {final_validation_score}")

# Submission-stage refit on full training data
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_cat_pred = X_full.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=600,
        learning_rate=0.04,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_full,
)

full_lgb_pred = X_full.skb.apply(vectorizer).skb.apply(
    LGBMClassifier(
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    ),
    y=y_full,
)

full_cat_learner = full_cat_pred.skb.make_learner(fitted=True)
full_lgb_learner = full_lgb_pred.skb.make_learner(fitted=True)

test_pred_cat = np.asarray(full_cat_learner.predict({"data": test_df})).ravel()
test_pred_lgb = np.asarray(full_lgb_learner.predict({"data": test_df})).ravel()

test_pred_avg = (test_pred_cat + test_pred_lgb) / 2.0
test_pred_bool = test_pred_avg > 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred_bool.astype(bool),
    }
)
submission["Transported"] = submission["Transported"].map({True: "True", False: "False"})
submission.to_csv("submission.csv", index=False)
