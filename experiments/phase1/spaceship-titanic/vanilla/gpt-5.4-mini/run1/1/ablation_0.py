import sys
import subprocess
import importlib.util
import warnings

warnings.filterwarnings("ignore")

if importlib.util.find_spec("catboost") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TRAIN_PATH = "./input/train.csv"

train = pd.read_csv(TRAIN_PATH)

def preprocess(df, use_engineered=True, use_cabin=True, use_group=True, use_spend=True):
    df = df.copy()

    if "PassengerId" in df.columns:
        df["PassengerId"] = df["PassengerId"].astype(str)

    if use_cabin and "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        df["Deck"] = cabin_split[0]
        df["Num"] = cabin_split[1]
        df["Side"] = cabin_split[2]
    else:
        df["Deck"] = np.nan
        df["Num"] = np.nan
        df["Side"] = np.nan

    if use_group and "PassengerId" in df.columns:
        df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
        df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
        df["GroupSize"] = df["GroupSize"].fillna(1)
        df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    else:
        df["Group"] = np.nan
        df["GroupSize"] = 1
        df["IsAlone"] = 1

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    if use_spend:
        for c in spend_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan
        df["TotalSpend"] = df[spend_cols].sum(axis=1)
        df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
        df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))
    else:
        for c in spend_cols:
            df[c] = np.nan
        df["TotalSpend"] = np.nan
        df["NoSpend"] = np.nan
        df["LogTotalSpend"] = np.nan

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    else:
        df["Age"] = np.nan

    df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

    numeric_cols = ["Age", "Num"] + spend_cols + ["TotalSpend", "LogTotalSpend", "GroupSize"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Missing").astype("category")

    drop_cols = ["Cabin", "Name", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

def evaluate_variant(name, use_cabin=True, use_group=True, use_spend=True, use_catboost=True):
    data = preprocess(train, use_cabin=use_cabin, use_group=use_group, use_spend=use_spend)

    target = "Transported"
    feature_cols = [c for c in data.columns if c != target]
    X = data[feature_cols].copy()
    y = data[target].astype(int).copy()

    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if use_catboost:
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            subsample=0.8,
            rsm=0.8
        )
        model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va), use_best_model=True)
        pred = model.predict(X_va).astype(int).reshape(-1)
    else:
        from sklearn.ensemble import RandomForestClassifier
        X_tr_enc = X_tr.copy()
        X_va_enc = X_va.copy()
        for c in cat_cols:
            X_tr_enc[c] = X_tr_enc[c].astype(str).fillna("Missing")
            X_va_enc[c] = X_va_enc[c].astype(str).fillna("Missing")
        X_all = pd.concat([X_tr_enc, X_va_enc], axis=0)
        X_all = pd.get_dummies(X_all, dummy_na=True)
        X_tr_enc = X_all.iloc[:len(X_tr_enc)]
        X_va_enc = X_all.iloc[len(X_tr_enc):]
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_tr_enc, y_tr)
        pred = model.predict(X_va_enc)

    score = accuracy_score(y_va, pred)
    print(f"{name}: validation accuracy = {score:.5f}")
    return score

base_score = evaluate_variant("Baseline", use_cabin=True, use_group=True, use_spend=True, use_catboost=True)

ablation_scores = {}

ablation_scores["No Cabin Features"] = evaluate_variant(
    "Ablation - No Cabin Features",
    use_cabin=False,
    use_group=True,
    use_spend=True,
    use_catboost=True
)

ablation_scores["No Group Features"] = evaluate_variant(
    "Ablation - No Group Features",
    use_cabin=True,
    use_group=False,
    use_spend=True,
    use_catboost=True
)

ablation_scores["No Spending Features"] = evaluate_variant(
    "Ablation - No Spending Features",
    use_cabin=True,
    use_group=True,
    use_spend=False,
    use_catboost=True
)

best_drop_name = None
best_drop_value = -1

for k, s in ablation_scores.items():
    drop = base_score - s
    print(f"{k}: score drop vs baseline = {drop:.5f}")
    if drop > best_drop_value:
        best_drop_value = drop
        best_drop_name = k

print(f"Most important removed component: {best_drop_name} (accuracy drop = {best_drop_value:.5f})")