
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

random_state = 42
np.random.seed(random_state)
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")


def preprocess_df(df):
    df = df.copy()

    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string")
        cabin_split = cabin.str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        else:
            df["CabinDeck"] = pd.NA
            df["CabinNum"] = np.nan
            df["CabinSide"] = pd.NA
        df = df.drop(columns=["Cabin"], errors="ignore")

    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype("string").str.split().str[-1]
        df = df.drop(columns=["Name"], errors="ignore")

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0)

    if "HomePlanet" in df.columns and "CabinDeck" in df.columns:
        df["PlanetDeck"] = df["HomePlanet"].astype("string") + "_" + df["CabinDeck"].astype("string")

    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 30, 50, np.inf],
            labels=["child", "teen", "young", "adult", "senior"],
        ).astype("string")

    return df


train_df = preprocess_df(train_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


import numpy as np
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_structural_features(df):
    out = df.copy()

    if "Name" in out.columns:
        out["Surname"] = (
            out["Name"].astype("string").fillna("").str.split(",").str[0].str.strip()
        )
    else:
        if "Surname" in out.columns:
            out["Surname"] = out["Surname"].astype("string").fillna("")
        else:
            out["Surname"] = ""

    if "Age" in out.columns:
        age = out["Age"]
        out["AgeGroup"] = np.select(
            [
                age.isna(),
                age < 16,
                age < 32,
                age < 48,
                age < 64,
            ],
            [
                "Unknown",
                "Child",
                "YoungAdult",
                "Adult",
                "MidAge",
            ],
            default="Senior",
        )
    elif "AgeGroup" not in out.columns:
        out["AgeGroup"] = "Unknown"

    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype("string").fillna("")
        out["CabinDeck"] = cabin.str.extract(r"([A-Za-z])", expand=False).fillna("Unknown")
        out["CabinNum"] = cabin.str.extract(r"(\d+)", expand=False).fillna("Unknown")
    else:
        if "CabinDeck" not in out.columns:
            out["CabinDeck"] = "Unknown"
        if "CabinNum" not in out.columns:
            out["CabinNum"] = "Unknown"

    if "Ticket" in out.columns:
        ticket = out["Ticket"].astype("string").fillna("")
        out["TicketPrefix"] = (
            ticket.str.replace(r"\d+", "", regex=True).str.replace(r"[./\s]+", "", regex=True).str.strip()
        )
        out["TicketPrefix"] = out["TicketPrefix"].replace("", "None")
    else:
        if "TicketPrefix" not in out.columns:
            out["TicketPrefix"] = "None"

    return out


data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_structural_features)
X_train = data_train_fe.drop(columns=[target_col, "PassengerId"], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=random_state,
    verbose=0,
)

lgbm_model = LGBMClassifier(
    random_state=random_state,
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
)

cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
lgbm_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(lgbm_model, y=y_train)

cat_learner = cat_pred.skb.make_learner(fitted=True)
lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)

valid_cat = np.asarray(cat_learner.predict({"data": valid_part})).reshape(-1).astype(float)
valid_lgbm = np.asarray(lgbm_learner.predict({"data": valid_part})).reshape(-1).astype(float)

valid_blend = 0.55 * valid_cat + 0.45 * valid_lgbm
valid_label = valid_blend >= 0.5

final_validation_score = accuracy_score(valid_part[target_col].astype(bool), valid_label)
print(f"Final Validation Performance: {final_validation_score}")
