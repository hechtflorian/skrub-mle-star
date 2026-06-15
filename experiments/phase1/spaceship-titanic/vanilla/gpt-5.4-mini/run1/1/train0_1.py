
import sys
import subprocess
import importlib.util
import warnings

warnings.filterwarnings("ignore")

if importlib.util.find_spec("catboost") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

if importlib.util.find_spec("lightgbm") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

def preprocess_catboost(df):
    df = df.copy()
    if "PassengerId" in df.columns:
        df["PassengerId"] = df["PassengerId"].astype(str)

    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        df["Deck"] = cabin_split[0]
        df["Num"] = cabin_split[1]
        df["Side"] = cabin_split[2]
    else:
        df["Deck"] = np.nan
        df["Num"] = np.nan
        df["Side"] = np.nan

    if "PassengerId" in df.columns:
        df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
    else:
        df["Group"] = np.nan

    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["GroupSize"] = df["GroupSize"].fillna(1)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    else:
        df["Age"] = np.nan

    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    numeric_cols = ["Age", "Num"] + spend_cols + ["TotalSpend", "LogTotalSpend", "GroupSize"]
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Missing").astype("category")

    drop_cols = ["Cabin", "Name", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df

def preprocess_lgbm(df):
    df = df.copy()
    df = df.replace({pd.NA: np.nan})

    cabin = df["Cabin"].astype("string").str.split("/", expand=True)
    df["Deck"] = cabin[0].astype("object")
    df["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2].astype("object")

    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))
    df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    for c in ["CryoSleep", "VIP"]:
        df[c] = df[c].astype("object")

    df = df.drop(columns=["Cabin", "Name", "PassengerId"])
    df = df.replace({pd.NA: np.nan})
    return df

train_cb = preprocess_catboost(train)
test_cb = preprocess_catboost(test)

X_cb = train_cb.drop(columns=["Transported"])
y = train_cb["Transported"].astype(int)
test_cb = test_cb.reindex(columns=X_cb.columns, fill_value=np.nan)
cat_cols_cb = [c for c in X_cb.columns if str(X_cb[c].dtype) == "category"]

X_tr, X_va, y_tr, y_va = train_test_split(X_cb, y, test_size=0.2, random_state=42, stratify=y)

cb_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=2500,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
    subsample=0.8,
    rsm=0.8
)

cb_model.fit(X_tr, y_tr, cat_features=cat_cols_cb, eval_set=(X_va, y_va), use_best_model=True)
cb_val_pred = cb_model.predict(X_va).astype(int).reshape(-1)
cb_val_score = accuracy_score(y_va, cb_val_pred)

best_iter = cb_model.get_best_iteration()
if best_iter is None or best_iter <= 0:
    best_iter = 2500

cb_final = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=best_iter,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
    subsample=0.8,
    rsm=0.8
)
cb_final.fit(X_cb, y, cat_features=cat_cols_cb)
cb_test_pred = cb_final.predict(test_cb).astype(int).reshape(-1)
cb_test_prob = cb_final.predict_proba(test_cb)[:, 1]

train_lgb = preprocess_lgbm(train)
test_lgb = preprocess_lgbm(test)

X_lgb = train_lgb.drop(columns=["Transported"])
test_lgb = test_lgb.reindex(columns=X_lgb.columns, fill_value=np.nan)

cat_cols_lgb = X_lgb.select_dtypes(include=["object", "bool", "category", "string"]).columns
num_cols_lgb = [c for c in X_lgb.columns if c not in cat_cols_lgb]

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols_lgb),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols_lgb),
    ]
)

lgb_model = Pipeline([
    ("pre", pre),
    ("clf", LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    ))
])

X_tr2, X_va2, y_tr2, y_va2 = train_test_split(X_lgb, y, test_size=0.2, random_state=42, stratify=y)
lgb_model.fit(X_tr2, y_tr2)
lgb_val_pred = lgb_model.predict(X_va2)
lgb_val_score = accuracy_score(y_va2, lgb_val_pred)

lgb_model.fit(X_lgb, y)
lgb_test_pred = lgb_model.predict(test_lgb)
lgb_test_prob = lgb_model.predict_proba(test_lgb)[:, 1]

final_validation_score = (cb_val_score + lgb_val_score) / 2.0
print(f"Final Validation Performance: {final_validation_score}")

ensemble_test_prob = (cb_test_prob + lgb_test_prob) / 2.0
ensemble_test_pred = (ensemble_test_prob >= 0.5).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": ensemble_test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
