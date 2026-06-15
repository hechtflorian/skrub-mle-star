
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

    # Cabin parsing
    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    # Light-weight name parsing: keep only coarse surname frequency, avoid brittle title-like tokens
    name_split = df["Name"].astype(str).str.split(" ", n=1, expand=True)
    df["LastName"] = name_split[1]
    surname = df["LastName"].fillna("Unknown").astype(str)
    surname_freq = surname.map(surname.value_counts())
    df["SurnameFreq"] = surname_freq
    df["SurnameFreqBin"] = pd.cut(
        df["SurnameFreq"].fillna(0),
        bins=[-1, 1, 2, 4, 9, 10**9],
        labels=False
    )

    # Existing age grouping
    age_bins = [-1, 12, 18, 30, 45, 60, 120]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=False)

    # Spending block
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))
    df["NoSpend"] = (df["TotalSpend"].fillna(0) == 0).astype(int)
    df["SpenderCount"] = (df[spend_cols].fillna(0) > 0).sum(axis=1)
    df["AllZeroSpend"] = (df[spend_cols].fillna(0).sum(axis=1) == 0).astype(int)

    # Interaction / cohort features
    if "HomePlanet" in df.columns and "Destination" in df.columns:
        df["HomePlanet_Destination"] = df["HomePlanet"].astype(str) + "_" + df["Destination"].astype(str)

    if "Deck" in df.columns and "Side" in df.columns:
        df["Deck_Side"] = df["Deck"].astype(str) + "_" + df["Side"].astype(str)

    if "AgeGroup" in df.columns and "CryoSleep" in df.columns:
        df["AgeGroup_CryoSleep"] = df["AgeGroup"].astype(str) + "_" + df["CryoSleep"].astype(str)

    if "HomePlanet" in df.columns and "CryoSleep" in df.columns:
        df["HomePlanet_CryoSleep"] = df["HomePlanet"].astype(str) + "_" + df["CryoSleep"].astype(str)

    if "Destination" in df.columns and "CryoSleep" in df.columns:
        df["Destination_CryoSleep"] = df["Destination"].astype(str) + "_" + df["CryoSleep"].astype(str)

    # Cabin / name availability indicators
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["NameKnown"] = df["Name"].notna().astype(int)

    # Keep deck grouping; useful as a coarse latent cohort feature
    df["DeckGroup"] = df["Deck"].replace({
        "A": "Upper", "B": "Upper", "C": "Upper", "T": "Upper",
        "D": "Middle", "E": "Middle", "F": "Lower", "G": "Lower"
    })

    # Drop raw text columns and brittle identity pieces
    df.drop(columns=["Cabin", "Name", "LastName"], inplace=True, errors="ignore")
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
