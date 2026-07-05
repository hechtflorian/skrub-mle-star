
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

random_state = 42

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "Transported"

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding must use train_part only for honest holdout validation
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

def preprocess_df(df):
    df = df.copy()
    if "PassengerId" in df.columns:
        parts = df["PassengerId"].astype(str).str.split("_", expand=True)
        if parts.shape[1] >= 2:
            df["PassengerGroup"] = parts[0]
            df["PassengerNum"] = pd.to_numeric(parts[1], errors="coerce")
        else:
            df["PassengerGroup"] = parts[0]
            df["PassengerNum"] = np.nan
    if "Cabin" in df.columns:
        cabin_parts = df["Cabin"].astype(str).str.split("/", expand=True)
        if cabin_parts.shape[1] >= 3:
            df["CabinDeck"] = cabin_parts[0]
            df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            df["CabinSide"] = cabin_parts[2]
    if "Name" in df.columns:
        df["NameLength"] = df["Name"].astype(str).str.len()
        df["Surname"] = df["Name"].astype(str).str.split().str[-1]
    if set(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]).intersection(df.columns):
        spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in df.columns]
        for c in spend_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["TotalSpend"] = df[spend_cols].sum(axis=1)
        df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 30, 50, np.inf],
            labels=["child", "teen", "young", "adult", "senior"],
        )
    if "CabinNum" in df.columns:
        df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")
    return df

# Keep skrub DataOps pipeline structure; fix prediction pipeline by ensuring the final step is a real estimator.
X_train_fe = X_train.skb.apply_func(preprocess_df)

vectorizer = skrub.TableVectorizer()
pred_lgbm = X_train_fe.skb.apply(vectorizer).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    ),
    y=y_train,
)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
valid_pred_lgbm = learner_lgbm.predict({"data": valid_part})

# Secondary leg kept in the same style; average only if it works cleanly
pred_cat = X_train_fe.skb.apply(vectorizer).skb.apply(
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
learner_cat = pred_cat.skb.make_learner(fitted=True)
valid_pred_cat = learner_cat.predict({"data": valid_part})

# Blend predictions from both learners
valid_proba_lgbm = np.asarray(valid_pred_lgbm)
valid_proba_cat = np.asarray(valid_pred_cat)
if valid_proba_lgbm.ndim > 1:
    valid_proba_lgbm = valid_proba_lgbm[:, 1]
if valid_proba_cat.ndim > 1:
    valid_proba_cat = valid_proba_cat[:, 1]

valid_pred = ((valid_proba_lgbm + valid_proba_cat) / 2.0) >= 0.5
valid_pred = pd.Series(valid_pred).map({True: True, False: False}).values

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
