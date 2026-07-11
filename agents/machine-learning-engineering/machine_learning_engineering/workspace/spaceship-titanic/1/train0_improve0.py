
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "-q"])
    import numpy as np

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"])
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation


train_path = os.path.join(".", "input", "train.csv")
test_path = os.path.join(".", "input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def engineer_features(df):
    df = df.copy()

    cabin_missing = df["Cabin"].isna()
    cabin_split = df["Cabin"].fillna("Unknown/0/U").str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
    group_size = df.groupby("Group")["Group"].transform("size")
    df["GroupSize"] = group_size
    df["IsSolo"] = (group_size == 1).astype(int)
    df["GroupSizeBin"] = pd.cut(
        group_size,
        bins=[0, 1, 2, 4, 8, np.inf],
        labels=["1", "2", "3-4", "5-8", "9+"],
        include_lowest=True
    ).astype(str)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    spend_filled = df[spend_cols].fillna(0)
    df["Spending"] = spend_filled.sum(axis=1)
    df["NoSpending"] = (df["Spending"] == 0).astype(int)
    df["HighSpending"] = (df["Spending"] > df["Spending"].median()).astype(int)
    df["SpendingPerPerson"] = df["Spending"] / df["GroupSize"].replace(0, 1)
    df["LuxurySpend"] = spend_filled["Spa"] + spend_filled["VRDeck"]
    df["BasicSpend"] = (
        spend_filled["RoomService"] + spend_filled["FoodCourt"] + spend_filled["ShoppingMall"]
    )

    df["CabinMissing"] = cabin_missing.astype(int)
    df["CabinNumMissing"] = df["CabinNum"].isna().astype(int)
    df["UnknownDeck"] = (df["Deck"] == "Unknown").astype(int)
    df["UnknownSide"] = (df["Side"] == "U").astype(int)
    df["DeckSide"] = df["Deck"].astype(str) + "_" + df["Side"].astype(str)

    if "CryoSleep" in df.columns:
        cryo = df["CryoSleep"].fillna(False).astype(int)
        df["CryoSleepFlag"] = cryo
        df["CryoAndNoSpending"] = ((cryo == 1) & (df["NoSpending"] == 1)).astype(int)

    if "VIP" in df.columns:
        vip = df["VIP"].fillna(False).astype(int)
        df["VIPFlag"] = vip
        df["VIPHighSpending"] = ((vip == 1) & (df["HighSpending"] == 1)).astype(int)

    if "Age" in df.columns:
        df["AgeMissing"] = df["Age"].isna().astype(int)
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 25, 40, 60, np.inf],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Mature", "Senior"]
        ).astype(str)

    drop_cols = [c for c in ["Cabin", "Name", "PassengerId"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


y = train["Transported"].astype(int)
X = engineer_features(train.drop(columns=["Transported"]))
X_test = engineer_features(test.copy())

cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_prep = preprocess.fit_transform(X_train)
X_valid_prep = preprocess.transform(X_valid)
X_test_prep = preprocess.transform(X_test)

clf = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=47,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

clf.fit(
    X_train_prep, y_train,
    eval_set=[(X_valid_prep, y_valid)],
    eval_metric="binary_logloss",
    callbacks=[early_stopping(100), log_evaluation(0)]
)

valid_pred = clf.predict(X_valid_prep)
valid_acc = accuracy_score(y_valid, valid_pred)

test_pred = clf.predict(X_test_prep).astype(bool)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {valid_acc:.6f}")
