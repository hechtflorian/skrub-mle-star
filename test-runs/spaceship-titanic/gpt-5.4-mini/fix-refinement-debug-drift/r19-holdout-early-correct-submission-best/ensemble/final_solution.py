import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

input_dir = "./input"
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))

def feature_engineer(df):
    df = df.copy()
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", n=1, expand=True)
        df["PassengerGroup"] = pd.to_numeric(pid[0], errors="coerce")
        df["PassengerNumber"] = pd.to_numeric(pid[1], errors="coerce")
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", n=2, expand=True)
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype(str).str.split(" ", n=1).str[-1]
        df["NameLen"] = df["Name"].astype(str).str.len()
    df["TotalSpending"] = (
        df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
        .fillna(0)
        .sum(axis=1)
    )
    df["SpendingFlag"] = (df["TotalSpending"] > 0).astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-np.inf, 12, 18, 30, 50, np.inf],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    return df

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(feature_engineer)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

pred_lgbm = X_train.skb.apply(
    vectorizer,
).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    ),
    y=y_train,
)

pred_cat = X_train.skb.apply(
    vectorizer,
).skb.apply(
    CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=random_state,
        verbose=0,
    ),
    y=y_train,
)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

def get_positive_score(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        return np.asarray(learner.predict({"data": df}), dtype=float).ravel()

valid_pred_lgbm = get_positive_score(learner_lgbm, valid_part)
valid_pred_cat = get_positive_score(learner_cat, valid_part)

best_score = -1.0
best_w = 0.5
for w in np.arange(0.2, 0.81, 0.1):
    blended_pred = w * valid_pred_lgbm + (1.0 - w) * valid_pred_cat
    final_pred = blended_pred >= 0.5
    score = accuracy_score(valid_part[target_col], final_pred)
    if score > best_score:
        best_score = score
        best_w = w

final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")

test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

data_full = skrub.var("data", train_df)
data_full = data_full.skb.apply_func(feature_engineer)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_lgbm = X_full.skb.apply(
    vectorizer,
).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    ),
    y=y_full,
)

full_pred_cat = X_full.skb.apply(
    vectorizer,
).skb.apply(
    CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=random_state,
        verbose=0,
    ),
    y=y_full,
)

full_learner_lgbm = full_pred_lgbm.skb.make_learner(fitted=True)
full_learner_cat = full_pred_cat.skb.make_learner(fitted=True)

test_pred_lgbm = get_positive_score(full_learner_lgbm, test_df)
test_pred_cat = get_positive_score(full_learner_cat, test_df)

test_blended_pred = best_w * test_pred_lgbm + (1.0 - best_w) * test_pred_cat
test_labels = np.where(test_blended_pred >= 0.5, True, False)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], target_col: test_labels})
submission.to_csv("./final/submission.csv", index=False)