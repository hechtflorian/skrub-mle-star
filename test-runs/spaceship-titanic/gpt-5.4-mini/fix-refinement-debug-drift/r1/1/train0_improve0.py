

import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

# Ensure CatBoost is available with a compatible wheel.
import catboost  # noqa: F401
from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"


@skrub.deferred
def add_features(df):
    out = df.copy()

    # Raw cabin pieces retained and cleaned, so TableVectorizer can encode them directly
    cabin_split = out["Cabin"].astype("string").str.split("/", expand=True)
    out["CabinDeck"] = cabin_split[0].astype("string")
    out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    out["CabinSide"] = cabin_split[2].astype("string")

    # Cheap structural cabin/group features
    out["Group"] = out["PassengerId"].astype("string").str.split("_").str[0]
    out["GroupSize"] = out.groupby("Group")["PassengerId"].transform("count")
    out["GroupHasMultipleCabins"] = out.groupby("Group")["Cabin"].transform("nunique").gt(1).astype(int)
    out["GroupHasMixedDeck"] = out.groupby("Group")["CabinDeck"].transform("nunique").gt(1).astype(int)
    out["CabinDeckSide"] = out["CabinDeck"].fillna("missing").astype("string") + "_" + out["CabinSide"].fillna("missing").astype("string")
    out["CabinDeckNum"] = out["CabinDeck"].fillna("missing").astype("string") + "_" + pd.cut(
        out["CabinNum"],
        bins=[-np.inf, 50, 250, 750, np.inf],
        labels=["low", "mid", "high", "very_high"],
    ).astype("string")

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["Spending"] = out[spending_cols].fillna(0).sum(axis=1)
    out["ZeroSpending"] = (out["Spending"] == 0).astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-np.inf, 12, 18, 30, 50, np.inf],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("string")

    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["CryoSleep_or_ZeroSpending"] = (
        (out["CryoSleep"].fillna(False).astype(bool)) | (out["ZeroSpending"] == 1)
    ).astype(int)

    out["LuxurySpending"] = out[["Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["NonLuxurySpending"] = out[["RoomService", "FoodCourt", "ShoppingMall"]].fillna(0).sum(axis=1)

    denom = out["Age"].replace(0, np.nan)
    out["SpendingPerAge"] = (out["Spending"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["CabinKnown"] = out["Cabin"].notna().astype(int)
    out["CabinNumLog"] = np.log1p(out["CabinNum"].fillna(-1).clip(lower=-1))
    out["CabinSideKnown"] = out["CabinSide"].notna().astype(int)

    # Name kept out on purpose per plan
    return out


# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps-native training graph: bind train_part only
data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)
X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

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

lgbm_pred = X_train.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_train)
cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)

# Honest validation
lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)
cat_learner = cat_pred.skb.make_learner(fitted=True)

valid_pred_lgbm = np.asarray(lgbm_learner.predict({"data": valid_part})).ravel()
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part})).ravel()

if valid_pred_lgbm.dtype != bool:
    valid_pred_lgbm = valid_pred_lgbm > 0.5
if valid_pred_cat.dtype != bool:
    valid_pred_cat = valid_pred_cat > 0.5

# Simplified ensemble: fixed weighted average rather than hard OR vote
valid_pred_proba = 0.65 * valid_pred_lgbm.astype(float) + 0.35 * valid_pred_cat.astype(float)
valid_pred_ensemble = valid_pred_proba >= 0.5
final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_ensemble)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: refit on full training data and predict test
data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)
X_full = data_full_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

full_pred_lgbm = X_full.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_full)
full_pred_cat = X_full.skb.apply(vectorizer).skb.apply(cat_model, y=y_full)

full_learner_lgbm = full_pred_lgbm.skb.make_learner(fitted=True)
full_learner_cat = full_pred_cat.skb.make_learner(fitted=True)

test_pred_lgbm = np.asarray(full_learner_lgbm.predict({"data": test_df})).ravel()
test_pred_cat = np.asarray(full_learner_cat.predict({"data": test_df})).ravel()

if test_pred_lgbm.dtype != bool:
    test_pred_lgbm = test_pred_lgbm > 0.5
if test_pred_cat.dtype != bool:
    test_pred_cat = test_pred_cat > 0.5

test_pred_proba = 0.65 * test_pred_lgbm.astype(float) + 0.35 * test_pred_cat.astype(float)
test_pred_ensemble = test_pred_proba >= 0.5

submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Transported": test_pred_ensemble.astype(bool)}
)
submission.to_csv("submission.csv", index=False)
