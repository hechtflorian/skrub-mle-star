
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

subprocess.run([sys.executable, "-m", "pip", "install", "catboost"], check=True, stdout=subprocess.DEVNULL)

from catboost import CatBoostClassifier

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"


def preprocess(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        df = df.drop(columns=["Cabin"], errors="ignore")
    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype("string").str.len()
        df = df.drop(columns=["Name"], errors="ignore")
    return df


train_df = preprocess(train_df)
test_df = preprocess(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

hgb_model = HistGradientBoostingClassifier(random_state=42)
cat_model = CatBoostClassifier(verbose=0, random_seed=42)

hgb_pred_graph = X_train.skb.apply(vectorizer).skb.apply(hgb_model, y=y_train)
cat_pred_graph = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(cat_model, y=y_train)

hgb_learner = hgb_pred_graph.skb.make_learner(fitted=True)
cat_learner = cat_pred_graph.skb.make_learner(fitted=True)

valid_pred_hgb = np.asarray(hgb_learner.predict({"data": valid_part})).ravel()
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part})).ravel()

if valid_pred_hgb.dtype != bool:
    valid_pred_hgb = valid_pred_hgb >= 0.5
if valid_pred_cat.dtype != bool:
    valid_pred_cat = valid_pred_cat >= 0.5

valid_pred_ens = (valid_pred_hgb.astype(int) + valid_pred_cat.astype(int)) >= 1
final_validation_score = accuracy_score(valid_part[target_col], valid_pred_ens)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_hgb_graph = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(hgb_model, y=y_full)
full_cat_graph = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(cat_model, y=y_full)

full_hgb_learner = full_hgb_graph.skb.make_learner(fitted=True)
full_cat_learner = full_cat_graph.skb.make_learner(fitted=True)

test_pred_hgb = np.asarray(full_hgb_learner.predict({"data": test_df})).ravel()
test_pred_cat = np.asarray(full_cat_learner.predict({"data": test_df})).ravel()

if test_pred_hgb.dtype != bool:
    test_pred_hgb = test_pred_hgb >= 0.5
if test_pred_cat.dtype != bool:
    test_pred_cat = test_pred_cat >= 0.5

test_pred = (test_pred_hgb.astype(int) + test_pred_cat.astype(int)) >= 1

submission = pd.DataFrame(
    {"PassengerId": pd.read_csv("./input/test.csv")["PassengerId"], "Transported": test_pred}
)
submission.to_csv("submission.csv", index=False)
