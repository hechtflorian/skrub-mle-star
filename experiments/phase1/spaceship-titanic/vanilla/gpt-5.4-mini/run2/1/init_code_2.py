
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def preprocess(df):
    df = df.copy()

    # Keep PassengerId only for final submission, not as a feature
    if "PassengerId" in df.columns:
        df = df.drop(columns=["PassengerId"])

    # Cabin features
    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    # Name features
    name_split = df["Name"].astype(str).str.split(" ", n=1, expand=True)
    df["FirstName"] = name_split[0]
    df["LastName"] = name_split[1]

    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["NameKnown"] = df["Name"].notna().astype(int)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["SpenderCount"] = (df[spend_cols] > 0).sum(axis=1)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 30, 45, 60, 120],
        labels=False
    )

    df["DeckGroup"] = df["Deck"].replace({
        "A": "Upper", "B": "Upper", "C": "Upper", "T": "Upper",
        "D": "Middle", "E": "Middle", "F": "Lower", "G": "Lower"
    })

    # Convert all boolean/object columns to numeric where appropriate
    bool_map = {
        True: 1, False: 0,
        "True": 1, "False": 0,
        "true": 1, "false": 0
    }
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
        elif df[col].dtype == "object":
            # try boolean mapping first, then keep as category codes
            mapped = df[col].map(bool_map)
            if mapped.notna().any():
                df[col] = mapped.astype("float")
            else:
                df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

    # Ensure numeric types for all remaining columns except target (handled outside)
    for col in df.columns:
        if col not in ["Transported"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.drop(columns=["Cabin", "Name", "FirstName", "LastName"], inplace=True, errors="ignore")
    return df

train = preprocess(train)
test = preprocess(test)

X = train.drop(columns=["Transported"])
y = train["Transported"].astype(int)

# Fill missing values
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category").cat.codes
    if col in test.columns and test[col].dtype == "object":
        test[col] = test[col].astype("category").cat.codes

# Align columns between train and test
test = test.reindex(columns=X.columns, fill_value=np.nan)

# Make sure everything is numeric for LightGBM
X = X.apply(pd.to_numeric, errors="coerce")
test = test.apply(pd.to_numeric, errors="coerce")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_error"
)

val_pred = model.predict(X_val).astype(int)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

test_pred = model.predict(test).astype(bool)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
