import os
import random
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")

train = pd.read_csv(train_path)

def feature_engineering(df):
    df = df.copy()

    cabin_split = df["Cabin"].fillna("NA/NA/NA").astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0].astype(str)
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].astype(str)

    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0].astype(str)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count").astype(int)

    df["Surname"] = df["Name"].fillna("NA").astype(str).str.split().str[-1].astype(str)
    df["HasSurname"] = (df["Surname"] != "NA").astype(int)
    df["SameSurnameGroupSize"] = df.groupby("Surname")["PassengerId"].transform("count").astype(int)

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]:
        if col in df.columns:
            df[col + "_isna"] = df[col].isna().astype(int)

    categorical_like = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "Surname"]
    for c in categorical_like:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    for c in ["CryoSleep", "VIP"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    return df

def prepare_data(df, use_feature_engineering=True, drop_engineered=None):
    if use_feature_engineering:
        df = feature_engineering(df)
    else:
        df = df.copy()

    y = df["Transported"].astype(int)
    X = df.drop(columns=["Transported"])

    if drop_engineered:
        X = X.drop(columns=[c for c in drop_engineered if c in X.columns])

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("NA").astype(str)

    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        if pd.isna(med):
            med = 0
        X[c] = X[c].fillna(med)

    return X, y, cat_cols

def run_experiment(name, use_feature_engineering=True, drop_engineered=None):
    X, y, cat_cols = prepare_data(train.copy(), use_feature_engineering=use_feature_engineering, drop_engineered=drop_engineered)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        random_seed=SEED,
        verbose=0,
        eval_metric="Accuracy"
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"{name}: Validation Accuracy = {val_acc:.6f}")
    return val_acc

baseline = run_experiment("Baseline")

ablation_no_feat_eng = run_experiment(
    "Ablation 1 - No feature engineering",
    use_feature_engineering=False
)

ablation_no_spending = run_experiment(
    "Ablation 2 - Remove spending features",
    use_feature_engineering=True,
    drop_engineered=["TotalSpending", "NoSpending", "RoomService_isna", "FoodCourt_isna", "ShoppingMall_isna", "Spa_isna", "VRDeck_isna"]
)

ablation_no_group_name = run_experiment(
    "Ablation 3 - Remove group/name features",
    use_feature_engineering=True,
    drop_engineered=["Group", "GroupSize", "Surname", "HasSurname", "SameSurnameGroupSize"]
)

results = {
    "Baseline": baseline,
    "No feature engineering": ablation_no_feat_eng,
    "No spending features": ablation_no_spending,
    "No group/name features": ablation_no_group_name,
}

best_ablation = min(results, key=lambda k: results[k])
worst_drop = baseline - min(ablation_no_feat_eng, ablation_no_spending, ablation_no_group_name)

print("\nSummary:")
for k, v in results.items():
    print(f"{k}: {v:.6f}")

print(f"\nMost important component (largest accuracy drop vs baseline): "
      f"{max(['No feature engineering', 'No spending features', 'No group/name features'], key=lambda k: baseline - results[k])}")
print(f"Largest observed drop: {worst_drop:.6f}")