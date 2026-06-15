
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
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

def prep(df):
    df = df.copy()

    # Standardize missing values so sklearn doesn't encounter pd.NA in object/string columns
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

    # Convert booleans to regular object strings to avoid pd.NA issues in preprocessing
    for c in ["CryoSleep", "VIP"]:
        df[c] = df[c].astype("object")

    df = df.drop(columns=["Cabin", "Name", "PassengerId"])
    df = df.replace({pd.NA: np.nan})
    return df

train_p = prep(train)
test_p = prep(test)

X = train_p.drop(columns=["Transported"])
y = train_p["Transported"].astype(int)

cat_cols = X.select_dtypes(include=["object", "bool", "category", "string"]).columns
num_cols = [c for c in X.columns if c not in cat_cols]

# Ensure all missing values are standard numpy NaN before sklearn sees them
X = X.replace({pd.NA: np.nan})
test_p = test_p.replace({pd.NA: np.nan})

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)

model = Pipeline([
    ("pre", pre),
    ("clf", LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    ))
])

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_tr, y_tr)
val_pred = model.predict(X_va)
final_validation_score = accuracy_score(y_va, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

model.fit(X, y)
test_pred = model.predict(test_p)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred.astype(bool)
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
