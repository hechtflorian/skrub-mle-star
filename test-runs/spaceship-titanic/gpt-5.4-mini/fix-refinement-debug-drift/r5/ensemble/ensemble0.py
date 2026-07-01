
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"

def preprocess_df(df):
    df = df.copy()

    if "PassengerId" in df.columns:
        df["PassengerGroup"] = df["PassengerId"].astype(str).str.split("_").str[0]
        df["PassengerNum"] = pd.to_numeric(
            df["PassengerId"].astype(str).str.split("_").str[1], errors="coerce"
        )
        df = df.drop(columns=["PassengerId"], errors="ignore")

    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
        df["CabinDeck"] = cabin_split[0]
        df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
        df["CabinSide"] = cabin_split[2]
        df = df.drop(columns=["Cabin"], errors="ignore")

    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype(str).str.len()
        df["NameWords"] = df["Name"].astype(str).str.split().str.len()
        df["Surname"] = df["Name"].astype(str).str.split().str[-1]
        df = df.drop(columns=["Name"], errors="ignore")

    numeric_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(col in df.columns for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]):
        df["Spent"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
        df["NoSpending"] = (df["Spent"].fillna(0) == 0).astype(int)

    if "Age" in df.columns:
        df["AgeBin"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 30, 45, 60, np.inf],
            labels=False,
        )

    if "CabinNum" in df.columns:
        df["CabinNumBin"] = pd.cut(
            df["CabinNum"],
            bins=[-np.inf, 1000, 2000, 3000, 4000, np.inf],
            labels=False,
        )

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

    return df

train_df_proc = preprocess_df(train_df)
test_df_proc = preprocess_df(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df_proc)), test_size=0.2, random_state=42
)
train_part = train_df_proc.iloc[train_idx].copy()
valid_part = train_df_proc.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

cat_model = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
    iterations=600,
    depth=7,
    learning_rate=0.03,
    l2_leaf_reg=4.0,
    subsample=0.8,
    rsm=0.85,
)

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

vectorizer = skrub.TableVectorizer()

cat_pred_graph = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
rf_pred_graph = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(rf_model, y=y_train)

cat_learner = cat_pred_graph.skb.make_learner(fitted=True)
rf_learner = rf_pred_graph.skb.make_learner(fitted=True)

valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}), dtype=bool)
valid_pred_rf = np.asarray(rf_learner.predict({"data": valid_part}), dtype=bool)

score_cat = accuracy_score(valid_part[target_col], valid_pred_cat)
score_rf = accuracy_score(valid_part[target_col], valid_pred_rf)

score_sum = score_cat + score_rf
if score_sum > 0:
    w_cat = score_cat / score_sum
    w_rf = score_rf / score_sum
else:
    w_cat, w_rf = 0.7, 0.3

if abs(score_cat - score_rf) < 1e-6:
    w_cat, w_rf = 0.7, 0.3

valid_pred_ens = (w_cat * valid_pred_cat.astype(float) + w_rf * valid_pred_rf.astype(float)) > 0.5
final_validation_score = accuracy_score(valid_part[target_col], valid_pred_ens)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df_proc)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_cat_pred_graph = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(cat_model, y=y_full)
full_rf_pred_graph = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(rf_model, y=y_full)

full_cat_learner = full_cat_pred_graph.skb.make_learner(fitted=True)
full_rf_learner = full_rf_pred_graph.skb.make_learner(fitted=True)

test_pred_cat = np.asarray(full_cat_learner.predict({"data": test_df_proc}), dtype=bool)
test_pred_rf = np.asarray(full_rf_learner.predict({"data": test_df_proc}), dtype=bool)

test_pred_ens = (w_cat * test_pred_cat.astype(float) + w_rf * test_pred_rf.astype(float)) > 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred_ens.astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
