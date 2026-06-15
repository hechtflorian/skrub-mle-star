
import os
import re
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def preprocess(df):
    df = df.copy()

    # --- Cabin-derived features: more robust handling of missing/unknown values ---
    cabin = df["Cabin"].astype("string")
    cabin_known = cabin.notna().astype(int)
    cabin_clean = cabin.fillna("Unknown/Unknown/Unknown")

    cabin_split = cabin_clean.str.split("/", expand=True)
    df["Deck"] = cabin_split[0].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    df["CabinKnown"] = cabin_known
    df["CabinUnknown"] = (df["Deck"] == "Unknown").astype(int)

    # Deck grouping kept categorical-friendly for CatBoost
    df["DeckGroup"] = df["Deck"].replace({
        "A": "Upper", "B": "Upper", "C": "Upper", "T": "Upper",
        "D": "Middle", "E": "Middle", "F": "Lower", "G": "Lower",
        "Unknown": "Unknown"
    })

    # --- Name-derived features: improved parsing with title extraction and family grouping ---
    name = df["Name"].astype("string")
    name_known = name.notna().astype(int)
    name_clean = name.fillna("Unknown Unknown")

    # Surname / given name parsing
    name_split = name_clean.str.split(" ", n=1, expand=True)
    df["LastName"] = name_split[0].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    df["FirstName"] = name_split[1].fillna("Unknown").replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    # Title extraction from the given name part when available
    title = df["FirstName"].str.extract(r"(?i)\b(Mr|Mrs|Miss|Master|Dr|Rev|Col|Major|Mlle|Ms|Mme|Capt|Sir|Lady|Countess|Jonkheer|Don|Dona)\b", expand=False)
    df["Title"] = title.fillna("Rare")

    # Better family grouping: surname + title + cabin deck for richer categorical signal
    family_id = df["LastName"].fillna("Unknown").astype(str)
    df["FamilySize"] = family_id.map(family_id.value_counts())

    # A small targeted tuning adjustment: family survival proxy-like grouping by surname frequency bucket
    df["FamilyGroup"] = pd.cut(
        df["FamilySize"].fillna(1),
        bins=[0, 1, 2, 4, 10, 10**9],
        labels=["Solo", "Duo", "Small", "Medium", "Large"],
        include_lowest=True
    ).astype(str)

    df["NameKnown"] = name_known
    df["NameUnknown"] = (df["LastName"] == "Unknown").astype(int)

    # --- Other features kept as secondary signals ---
    age_bins = [-1, 12, 18, 30, 45, 60, 120]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=False)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["SpenderCount"] = (df[spend_cols] > 0).sum(axis=1)

    # Preserve enriched categorical features for CatBoost
    df["SurnameTitle"] = df["LastName"].astype(str) + "_" + df["Title"].astype(str)
    df["DeckSide"] = df["Deck"].astype(str) + "_" + df["Side"].astype(str)

    df.drop(columns=["Cabin", "Name"], inplace=True, errors="ignore")
    return df


train = preprocess(train)
test = preprocess(test)

X = train.drop(columns=["Transported"])
y = train["Transported"].astype(int)

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype(str).fillna("Missing")
    test[col] = test[col].astype(str).fillna("Missing")

for col in X.columns:
    if col not in cat_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
    cat_features=[X.columns.get_loc(c) for c in cat_cols],
    eval_set=(X_val, y_val),
    use_best_model=True
)

val_pred = model.predict(X_val).astype(int).ravel()
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

test_pred = model.predict(test).astype(bool).ravel()

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
