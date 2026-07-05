
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

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def add_features(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin = df["Cabin"].fillna("missing").astype(str).str.split("/", expand=True)
        if cabin.shape[1] >= 3:
            df["CabinDeck"] = cabin[0]
            df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
            df["CabinSide"] = cabin[2]
        else:
            df["CabinDeck"] = np.nan
            df["CabinNum"] = np.nan
            df["CabinSide"] = np.nan
    if "Name" in df.columns:
        name_split = df["Name"].fillna("missing").astype(str).str.split(" ", n=1, expand=True)
        df["Surname"] = name_split[1] if name_split.shape[1] > 1 else np.nan
    spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in df.columns]
    if spend_cols:
        spend = df[spend_cols].fillna(0)
        df["TotalSpend"] = spend.sum(axis=1)
        df["NoSpend"] = (spend.sum(axis=1) == 0).astype(int)
    return df

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

X_train_fe = X_train.skb.apply_func(add_features)

vectorizer = skrub.TableVectorizer()
cat_model = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)
lgbm_model = LGBMClassifier(
    random_state=random_state,
    n_estimators=300,
    verbose=-1,
)

pred_cat = X_train_fe.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
pred_lgbm = X_train_fe.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_train)

learner_cat = pred_cat.skb.make_learner(fitted=True)
learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)

valid_pred_cat = learner_cat.predict({"data": valid_part})
valid_pred_lgbm = learner_lgbm.predict({"data": valid_part})

valid_pred_cat = np.asarray(valid_pred_cat).ravel()
valid_pred_lgbm = np.asarray(valid_pred_lgbm).ravel()

if valid_pred_cat.dtype != np.float64 and valid_pred_cat.dtype != np.float32:
    valid_pred_cat = (valid_pred_cat == True).astype(float)
if valid_pred_lgbm.dtype != np.float64 and valid_pred_lgbm.dtype != np.float32:
    valid_pred_lgbm = (valid_pred_lgbm == True).astype(float)

valid_pred = (0.5 * valid_pred_cat + 0.5 * valid_pred_lgbm) >= 0.5
valid_pred = pd.Series(valid_pred).map({True: True, False: False}).values

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
