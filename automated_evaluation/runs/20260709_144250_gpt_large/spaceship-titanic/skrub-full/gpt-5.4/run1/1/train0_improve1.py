
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

target_col = "Transported"
random_state = 42

train_df = pd.read_csv(TRAIN_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


data_train = skrub.var("data", train_part)

def add_compact_features(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin_split = out["Cabin"].astype("string").str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            out["CabinDeck"] = cabin_split[0]
            out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            out["CabinSide"] = cabin_split[2]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_spend_cols = [c for c in spend_cols if c in out.columns]
    if existing_spend_cols:
        out["TotalSpend"] = out[existing_spend_cols].fillna(0).sum(axis=1)
    else:
        out["TotalSpend"] = 0.0

    out["ZeroSpend"] = (out["TotalSpend"] <= 0).astype(int)

    if "PassengerId" in out.columns:
        pid_split = out["PassengerId"].astype("string").str.split("_", n=1, expand=True)
        if pid_split.shape[1] >= 2:
            out["GroupId"] = pid_split[0]
            out["WithinGroupNum"] = pd.to_numeric(pid_split[1], errors="coerce")

    cryo_num = (
        out["CryoSleep"].astype(str).map({"True": 1, "False": 0, "True ": 1, "False ": 0}).fillna(0).astype(int)
        if "CryoSleep" in out.columns
        else 0
    )
    vip_num = (
        out["VIP"].astype(str).map({"True": 1, "False": 0, "True ": 1, "False ": 0}).fillna(0).astype(int)
        if "VIP" in out.columns
        else 0
    )

    out["CryoSleep_ZeroSpend"] = cryo_num * out["ZeroSpend"]
    out["VIP_TotalSpend"] = vip_num * out["TotalSpend"]

    if "Age" in out.columns:
        out["AgeBand"] = pd.cut(
            pd.to_numeric(out["Age"], errors="coerce"),
            bins=[-np.inf, 12, 18, 30, 45, 60, np.inf],
            labels=["Child", "Teen", "YoungAdult", "Adult", "MidAge", "Senior"],
        )

    out = out.drop(columns=["Name", "Cabin"], errors="ignore")
    return out

data_train_fe = data_train.skb.apply_func(add_compact_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality="drop",
)

lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

predictor_lgbm = (
    X_train
    .skb.apply(vectorizer_lgbm)
    .skb.apply(lgbm_model, y=y_train)
)

learner_lgbm = predictor_lgbm.skb.make_learner(fitted=True)

valid_pred_lgbm = pd.Series(learner_lgbm.predict({"data": valid_part}), index=valid_part.index)

valid_true = pd.Series(valid_part[target_col], index=valid_part.index).astype(str).map({"True": True, "False": False}).fillna(valid_part[target_col])
valid_pred_lgbm_bool = valid_pred_lgbm.astype(str).map({"True": True, "False": False}).fillna(valid_pred_lgbm)
valid_pred_lgbm_num = pd.Series(valid_pred_lgbm_bool, index=valid_part.index).astype(bool).astype(int)

ensemble_vote = valid_pred_lgbm_num.astype(bool)


final_validation_score = accuracy_score(valid_true.astype(bool), ensemble_vote)
print(f"Final Validation Performance: {final_validation_score}")
