

import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

import catboost  # noqa: F401
from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
import skrub
import skrub.selectors as s
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"


@skrub.deferred
def add_features(df):
    out = df.copy()

    cabin_split = out["Cabin"].astype("string").str.split("/", expand=True)
    out["CabinDeck"] = cabin_split[0]
    out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    out["CabinSide"] = cabin_split[2]
    out["CabinDeckSide"] = (
        out["CabinDeck"].astype("string").fillna("missing")
        + "_"
        + out["CabinSide"].astype("string").fillna("missing")
    )

    out["Group"] = out["PassengerId"].astype("string").str.split("_").str[0]
    out["GroupSize"] = out.groupby("Group")["PassengerId"].transform("count")

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        out[f"{col}_is_missing"] = out[col].isna().astype(int)
    out["Spending"] = out[spending_cols].fillna(0).sum(axis=1)
    out["ZeroSpending"] = (out["Spending"] == 0).astype(int)

    out["Age_is_missing"] = out["Age"].isna().astype(int)
    out["CabinNum_is_missing"] = out["CabinNum"].isna().astype(int)
    out["VIP_is_missing"] = out["VIP"].isna().astype(int)
    out["CryoSleep_is_missing"] = out["CryoSleep"].isna().astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-np.inf, 12, 18, 30, 50, np.inf],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("string")

    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["HasName"] = out["Name"].notna().astype(int)

    cryo = out["CryoSleep"].fillna(False).astype(bool)
    out["CryoSleep_or_ZeroSpending"] = (cryo | (out["ZeroSpending"] == 1)).astype(int)

    out["LuxurySpending"] = out[["Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["NonLuxurySpending"] = out[["RoomService", "FoodCourt", "ShoppingMall"]].fillna(0).sum(axis=1)

    out["RoomService_is_missing"] = out["RoomService"].isna().astype(int)
    out["FoodCourt_is_missing"] = out["FoodCourt"].isna().astype(int)
    out["ShoppingMall_is_missing"] = out["ShoppingMall"].isna().astype(int)
    out["Spa_is_missing"] = out["Spa"].isna().astype(int)
    out["VRDeck_is_missing"] = out["VRDeck"].isna().astype(int)

    denom = out["Age"].replace(0, np.nan)
    out["SpendingPerAge"] = (out["Spending"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)
X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

high_card_selector = s.cols(
    "CabinDeck",
    "CabinSide",
    "CabinDeckSide",
    "Group",
    "AgeGroup",
    "Name",
    "PassengerId",
)

numeric_scale_selector = s.cols(
    "Age",
    "CabinNum",
    "Spending",
    "LuxurySpending",
    "NonLuxurySpending",
    "SpendingPerAge",
    "GroupSize",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
)

default_selector = s.all() - high_card_selector - numeric_scale_selector

high_card_path = X_train.skb.select(high_card_selector).skb.apply(
    skrub.TableVectorizer(high_cardinality=skrub.GapEncoder())
)
numeric_scale_path = X_train.skb.select(numeric_scale_selector).skb.apply(
    SimpleImputer(strategy="median")
).skb.apply(
    RobustScaler(with_centering=False)
)
default_path = X_train.skb.select(default_selector).skb.apply(skrub.TableVectorizer())
X_train_prepared = default_path.skb.concat([high_card_path, numeric_scale_path], axis=1)

lgbm_model = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    iterations=350,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
)

lgbm_pred = X_train_prepared.skb.apply(lgbm_model, y=y_train)
cat_pred = X_train_prepared.skb.apply(cat_model, y=y_train)

lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)
cat_learner = cat_pred.skb.make_learner(fitted=True)

valid_pred_lgbm = np.asarray(lgbm_learner.predict({"data": valid_part})).ravel()
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part})).ravel()

if valid_pred_lgbm.dtype != bool:
    valid_pred_lgbm = valid_pred_lgbm > 0.5
if valid_pred_cat.dtype != bool:
    valid_pred_cat = valid_pred_cat > 0.5

valid_pred_ensemble = (0.6 * valid_pred_lgbm.astype(float) + 0.4 * valid_pred_cat.astype(float)) >= 0.5
final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_ensemble)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)
X_full = data_full_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

high_card_path_full = X_full.skb.select(high_card_selector).skb.apply(
    skrub.TableVectorizer(high_cardinality=skrub.GapEncoder())
)
numeric_scale_path_full = X_full.skb.select(numeric_scale_selector).skb.apply(
    SimpleImputer(strategy="median")
).skb.apply(
    RobustScaler(with_centering=False)
)
default_path_full = X_full.skb.select(default_selector).skb.apply(skrub.TableVectorizer())
X_full_prepared = default_path_full.skb.concat([high_card_path_full, numeric_scale_path_full], axis=1)

full_pred_lgbm = X_full_prepared.skb.apply(lgbm_model, y=y_full)
full_pred_cat = X_full_prepared.skb.apply(cat_model, y=y_full)

full_learner_lgbm = full_pred_lgbm.skb.make_learner(fitted=True)
full_learner_cat = full_pred_cat.skb.make_learner(fitted=True)

test_pred_lgbm = np.asarray(full_learner_lgbm.predict({"data": test_df})).ravel()
test_pred_cat = np.asarray(full_learner_cat.predict({"data": test_df})).ravel()

if test_pred_lgbm.dtype != bool:
    test_pred_lgbm = test_pred_lgbm > 0.5
if test_pred_cat.dtype != bool:
    test_pred_cat = test_pred_cat > 0.5

test_pred_ensemble = (0.6 * test_pred_lgbm.astype(float) + 0.4 * test_pred_cat.astype(float)) >= 0.5

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Transported": test_pred_ensemble.astype(bool)}
)
submission.to_csv("submission.csv", index=False)
