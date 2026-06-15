import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")

train = pd.read_csv(train_path)

def preprocess(df):
    df = df.copy()

    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    name_split = df["Name"].astype(str).str.split(" ", n=1, expand=True)
    df["FirstName"] = name_split[0]
    df["LastName"] = name_split[1]

    family_id = df["LastName"].fillna("Unknown").astype(str)
    fam_counts = family_id.map(family_id.value_counts())
    df["FamilySize"] = fam_counts

    age_bins = [-1, 12, 18, 30, 45, 60, 120]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=False)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["SpenderCount"] = (df[spend_cols] > 0).sum(axis=1)

    df["DeckGroup"] = df["Deck"].replace({
        "A": "Upper", "B": "Upper", "C": "Upper", "T": "Upper",
        "D": "Middle", "E": "Middle", "F": "Lower", "G": "Lower"
    })

    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["NameKnown"] = df["Name"].notna().astype(int)

    df.drop(columns=["Cabin", "Name", "FirstName", "LastName"], inplace=True, errors="ignore")
    return df

train = preprocess(train)

X = train.drop(columns=["Transported"])
y = train["Transported"].astype(int)

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype(str).fillna("Missing")

for col in X.columns:
    if col not in cat_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def fit_and_eval(X_train, y_train, X_val, y_val, cat_cols, desc):
    model = CatBoostClassifier(
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
        allow_writing_files=False
    )
    model.fit(
        X_train,
        y_train,
        cat_features=[X_train.columns.get_loc(c) for c in cat_cols],
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    pred = model.predict(X_val).astype(int).ravel()
    acc = accuracy_score(y_val, pred)
    print(f"{desc}: Validation Accuracy = {acc:.5f}")
    return acc

# Baseline
baseline_acc = fit_and_eval(X_train, y_train, X_val, y_val, cat_cols, "Baseline")

# Ablation 1: remove engineered spending features
spend_feats = ["TotalSpend", "NoSpend", "SpenderCount"]
X_train_no_spend = X_train.drop(columns=spend_feats, errors="ignore")
X_val_no_spend = X_val.drop(columns=spend_feats, errors="ignore")
cat_cols_no_spend = [c for c in cat_cols if c not in spend_feats]

acc_no_spend = fit_and_eval(
    X_train_no_spend, y_train, X_val_no_spend, y_val, cat_cols_no_spend,
    "Ablation 1 (remove spending features)"
)

# Ablation 2: remove cabin/name-derived features
cab_name_feats = ["Deck", "Num", "Side", "FamilySize", "AgeGroup", "DeckGroup", "CabinKnown", "NameKnown"]
X_train_no_cabname = X_train.drop(columns=cab_name_feats, errors="ignore")
X_val_no_cabname = X_val.drop(columns=cab_name_feats, errors="ignore")
cat_cols_no_cabname = [c for c in cat_cols if c not in cab_name_feats]

acc_no_cabname = fit_and_eval(
    X_train_no_cabname, y_train, X_val_no_cabname, y_val, cat_cols_no_cabname,
    "Ablation 2 (remove cabin/name-derived features)"
)

# Report impact
drop_spend = baseline_acc - acc_no_spend
drop_cabname = baseline_acc - acc_no_cabname

print("\nPerformance impact vs baseline:")
print(f"Remove spending features: {drop_spend:+.5f}")
print(f"Remove cabin/name-derived features: {drop_cabname:+.5f}")

if drop_spend > drop_cabname:
    print("Most important part: spending feature engineering contributes the most to performance.")
elif drop_cabname > drop_spend:
    print("Most important part: cabin/name-derived feature engineering contributes the most to performance.")
else:
    print("Both parts contribute similarly to performance.")