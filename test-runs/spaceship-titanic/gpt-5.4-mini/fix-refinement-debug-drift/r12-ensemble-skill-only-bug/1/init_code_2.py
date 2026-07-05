
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def prep_columns(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", expand=True)
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"])

    if "Name" in out.columns:
        out["NameLength"] = out["Name"].astype("string").str.len()
        out["NameTokens"] = out["Name"].astype("string").str.split().str.len()
        out = out.drop(columns=["Name"])

    bool_cols = [c for c in ["CryoSleep", "VIP"] if c in out.columns]
    for c in bool_cols:
        out[c] = out[c].astype("string").map({"True": 1, "False": 0}).astype(float)

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

pred_graph = X_train.skb.apply_func(prep_columns)

def make_model_df(df):
    out = df.copy()
    if "PassengerId" in out.columns:
        out["Group"] = out["PassengerId"].astype("string").str.split("_").str[0]
        out["Person"] = out["PassengerId"].astype("string").str.split("_").str[1]
    return out

pred_graph = pred_graph.skb.apply_func(make_model_df).skb.apply(skrub.TableVectorizer()).skb.apply(
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

val_learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = pd.Series(valid_pred).astype(bool)
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_graph = X_full.skb.apply_func(prep_columns)
full_pred_graph = full_pred_graph.skb.apply_func(make_model_df).skb.apply(skrub.TableVectorizer()).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    ),
    y=y_full,
)

full_learner = full_pred_graph.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = pd.Series(test_pred).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred.map({True: "True", False: "False"}).values,
    }
)
submission.to_csv("submission.csv", index=False)
