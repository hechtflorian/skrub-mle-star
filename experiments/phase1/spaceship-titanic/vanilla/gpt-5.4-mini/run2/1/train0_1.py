
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

    if "PassengerId" in df.columns:
        df = df.drop(columns=["PassengerId"])

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

    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["NameKnown"] = df["Name"].notna().astype(int)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["SpenderCount"] = (df[spend_cols] > 0).sum(axis=1)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[-1, 12, 18, 30, 45, 60, 120], labels=False)

    df["DeckGroup"] = df["Deck"].replace({
        "A": "Upper", "B": "Upper", "C": "Upper", "T": "Upper",
        "D": "Middle", "E": "Middle", "F": "Lower", "G": "Lower"
    })

    bool_map = {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0}
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
        elif df[col].dtype == "object":
            mapped = df[col].map(bool_map)
            if mapped.notna().any():
                df[col] = mapped.astype("float")
            else:
                df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

    for col in df.columns:
        if col not in ["Transported"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.drop(columns=["Cabin", "Name", "FirstName", "LastName"], inplace=True, errors="ignore")
    return df

train = preprocess(train)
test = preprocess(test)

X = train.drop(columns=["Transported"])
y = train["Transported"].astype(int)

test = test.reindex(columns=X.columns, fill_value=np.nan)

X = X.apply(pd.to_numeric, errors="coerce")
test = test.apply(pd.to_numeric, errors="coerce")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_model = CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0,
    allow_writing_files=False
)

lgb_model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_error")

cat_val_prob = cat_model.predict_proba(X_val)[:, 1]
lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
ensemble_val_prob = 0.5 * cat_val_prob + 0.5 * lgb_val_prob
val_pred = (ensemble_val_prob >= 0.5).astype(int)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

cat_test_prob = cat_model.predict_proba(test)[:, 1]
lgb_test_prob = lgb_model.predict_proba(test)[:, 1]
ensemble_test_prob = 0.5 * cat_test_prob + 0.5 * lgb_test_prob
test_pred = (ensemble_test_prob >= 0.5)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
