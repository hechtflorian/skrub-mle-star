
import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def preprocess(df):
    df = df.copy()

    # Cabin parsing
    cabin = df["Cabin"].astype(str)
    cabin_split = cabin.str.split("/", expand=True)
    df["Deck"] = cabin_split[0].replace("nan", np.nan)
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].replace("nan", np.nan)

    # Name parsing
    name = df["Name"].astype(str)
    name_split = name.str.split(" ", n=1, expand=True)
    df["FirstName"] = name_split[0].replace("nan", np.nan)
    df["LastName"] = name_split[1].replace("nan", np.nan)

    # Family features
    df["LastNameFilled"] = df["LastName"].fillna("Unknown")
    family_sizes = df["LastNameFilled"].value_counts()
    df["FamilySize"] = df["LastNameFilled"].map(family_sizes).astype(float)

    # Age features
    df["IsChild"] = (df["Age"] <= 12).astype(float)
    df["IsSenior"] = (df["Age"] >= 60).astype(float)
    df["AgeMissing"] = df["Age"].isna().astype(float)

    # Spend features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(float)
    df["SpendVar"] = df[spend_cols].var(axis=1).fillna(0.0)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"])

    # Missing indicators
    df["CabinMissing"] = df["Cabin"].isna().astype(float)
    df["NameMissing"] = df["Name"].isna().astype(float)

    # Simplify some categoricals
    df["DeckGroup"] = df["Deck"].replace({
        "A": "Upper", "B": "Upper", "C": "Upper", "T": "Upper",
        "D": "Middle", "E": "Middle", "F": "Lower", "G": "Lower"
    })

    # Useful interaction-style features
    df["CryoSleep"] = df["CryoSleep"].astype("object")
    df["VIP"] = df["VIP"].astype("object")

    # Clean up text helper columns
    df.drop(columns=["Cabin", "Name", "LastNameFilled"], inplace=True, errors="ignore")
    return df

train = preprocess(train)
test = preprocess(test)

y = train["Transported"].astype(int)
X = train.drop(columns=["Transported"])

# Match test columns to train columns after preprocessing
X, test = X.align(test, join="left", axis=1)

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Convert categoricals to string and fill missing
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("Missing")
    test[col] = test[col].astype(str).fillna("Missing")

# Numeric columns: coerce and impute
num_cols = [c for c in X.columns if c not in cat_cols]
for col in num_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")

num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])

# Train/validation split for a quick sanity check
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_idx = [X.columns.get_loc(c) for c in cat_cols]

def build_model(seed=42):
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
        od_type="Iter",
        od_wait=50
    )

# Train a single strong model to avoid timeout
model = build_model(seed=42)
model.fit(
    X_train,
    y_train,
    cat_features=cat_idx,
    eval_set=(X_val, y_val),
    use_best_model=True
)

val_pred = model.predict(X_val).astype(int).reshape(-1)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

# Fit on full data and predict test
full_model = build_model(seed=42)
full_model.fit(X, y, cat_features=cat_idx, verbose=0)

test_pred = full_model.predict(test).astype(int).reshape(-1).astype(bool)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)
