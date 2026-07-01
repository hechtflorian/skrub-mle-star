
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))

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

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df_proc)), test_size=0.2, random_state=42
)
train_part = train_df_proc.iloc[train_idx].copy()
valid_part = train_df_proc.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

catboost_variants = {
    "d7_lr0.03_l2_4_sub0.8_rsm0.85": dict(
        depth=7, learning_rate=0.03, l2_leaf_reg=4.0, subsample=0.8, rsm=0.85
    ),
    "d8_lr0.03_l2_4_sub0.8_rsm0.85": dict(
        depth=8, learning_rate=0.03, l2_leaf_reg=4.0, subsample=0.8, rsm=0.85
    ),
    "d7_lr0.05_l2_4_sub0.8_rsm0.85": dict(
        depth=7, learning_rate=0.05, l2_leaf_reg=4.0, subsample=0.8, rsm=0.85
    ),
    "d8_lr0.05_l2_4_sub0.8_rsm0.85": dict(
        depth=8, learning_rate=0.05, l2_leaf_reg=4.0, subsample=0.8, rsm=0.85
    ),
}
cat_model = skrub.choose_from(
    {
        k: CatBoostClassifier(
            loss_function="Logloss",
            random_seed=42,
            verbose=0,
            iterations=300,
            **v,
        )
        for k, v in catboost_variants.items()
    },
    name="catboost_variant",
)

data_train_2 = skrub.var("data2", train_part)
X_train_2 = data_train_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_2 = data_train_2[target_col].skb.mark_as_y()

rf_variants = {
    "n400_leaf2": dict(n_estimators=400, min_samples_leaf=2),
    "n600_leaf2": dict(n_estimators=600, min_samples_leaf=2),
    "n400_leaf1": dict(n_estimators=400, min_samples_leaf=1),
    "n600_leaf1": dict(n_estimators=600, min_samples_leaf=1),
}
rf_model = skrub.choose_from(
    {
        k: __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            **v,
        )
        for k, v in rf_variants.items()
    },
    name="rf_variant",
)

cat_pred_graph = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(cat_model, y=y_train)
rf_pred_graph = X_train_2.skb.apply(skrub.TableVectorizer()).skb.apply(rf_model, y=y_train_2)

pred = 0.7 * cat_pred_graph + 0.3 * rf_pred_graph
search = pred.skb.make_randomized_search(n_iter=4, n_jobs=1, random_state=42, fitted=True)
search.fit({"data": train_part})

valid_pred_cat = np.asarray(search.best_learner_.predict({"data": valid_part}), dtype=float)
valid_pred_rf = np.asarray(search.best_learner_.predict({"data2": valid_part}), dtype=float)
valid_pred_ens = 0.7 * valid_pred_cat + 0.3 * valid_pred_rf
valid_pred_binary = valid_pred_ens > 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred_binary)
print(f"Final Validation Performance: {final_validation_score}")

best_params = {}
bp = search.best_params_
best_params["catboost_variant"] = next(
    k for k, v in catboost_variants.items()
    if all(bp_val == v[name] for name, bp_val in zip(["depth", "learning_rate", "l2_leaf_reg", "subsample", "rsm"], [v["depth"], v["learning_rate"], v["l2_leaf_reg"], v["subsample"], v["rsm"]]))
)
best_params["rf_variant"] = next(
    k for k, v in rf_variants.items()
    if all(bp_val == v[name] for name, bp_val in zip(["n_estimators", "min_samples_leaf"], [v["n_estimators"], v["min_samples_leaf"]]))
)
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
