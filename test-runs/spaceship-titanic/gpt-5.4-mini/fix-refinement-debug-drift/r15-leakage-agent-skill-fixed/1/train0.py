
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)

def feature_engineering(df):
    df = df.copy()
    cabin = df["Cabin"].astype(str).str.split("/", expand=True)
    if cabin.shape[1] == 3:
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
    else:
        df["CabinDeck"] = np.nan
        df["CabinNum"] = np.nan
        df["CabinSide"] = np.nan
    df["NameLength"] = df["Name"].astype(str).str.len()
    return df

data_train = data_train.skb.apply_func(feature_engineering)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_cat = skrub.TableVectorizer()
vectorizer_lgbm = skrub.TableVectorizer()

model_cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

model_lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=random_state,
    n_jobs=1,
    verbose=-1,
)

pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(model_cat, y=y_train)
pred_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(model_lgbm, y=y_train)

learner_cat = pred_cat.skb.make_learner(fitted=True)
learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)

def get_pred(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    except Exception:
        pred = learner.predict({"data": df})
        pred = np.asarray(pred)
        if pred.dtype == bool:
            return pred.astype(int)
        return pred.ravel()

cat_valid_raw = get_pred(learner_cat, valid_part)
lgbm_valid_raw = get_pred(learner_lgbm, valid_part)

valid_pred = 0.5 * cat_valid_raw + 0.5 * lgbm_valid_raw
valid_pred = valid_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
