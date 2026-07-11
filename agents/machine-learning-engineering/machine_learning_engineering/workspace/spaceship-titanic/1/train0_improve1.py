
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from lightgbm import LGBMClassifier
except ModuleNotFoundError as e:
    missing_module = str(e).split("'")[1]
    subprocess.check_call([sys.executable, "-m", "pip", "install", missing_module])
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from lightgbm import LGBMClassifier


train_path = os.path.join(".", "input", "train.csv")
test_path = os.path.join(".", "input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def engineer_features(df):
    df = df.copy()

    cabin_split = df["Cabin"].fillna("Unknown/0/U").str.split("/", expand=True)
    df["Deck"] = cabin_split[0].fillna("Unknown")
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].fillna("U")

    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
    df["Surname"] = (
        df["Name"]
        .fillna("Unknown Unknown")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.split(" ")
        .str[-1]
        .fillna("Unknown")
    )

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["Spending"] = df[spend_cols].fillna(0).sum(axis=1)
    df["NoSpending"] = (df["Spending"] == 0).astype(int)

    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["CabinNumMissing"] = df["CabinNum"].isna().astype(int)

    group_size = df.groupby("Group")["Group"].transform("size")
    surname_size = df.groupby("Surname")["Surname"].transform("size")

    df["GroupSize"] = group_size
    df["SurnameSize"] = surname_size
    df["SurnameGroupMatch"] = (df["GroupSize"] == df["SurnameSize"]).astype(int)
    df["SurnameInMultipleGroups"] = (
        df.groupby("Surname")["Group"].transform("nunique") > 1
    ).astype(int)

    df["GroupDeckNunique"] = df.groupby("Group")["Deck"].transform("nunique")
    df["GroupSideNunique"] = df.groupby("Group")["Side"].transform("nunique")
    df["GroupCabinNumNunique"] = df.groupby("Group")["CabinNum"].transform(
        lambda x: x.nunique(dropna=True)
    )

    df["GroupDeckConsistent"] = (df["GroupDeckNunique"] == 1).astype(int)
    df["GroupSideConsistent"] = (df["GroupSideNunique"] == 1).astype(int)
    df["GroupCabinNumConsistent"] = (df["GroupCabinNumNunique"] <= 1).astype(int)

    df["CabinNumRankPct"] = df["CabinNum"].rank(method="average", pct=True)
    df["SpendingRankPct"] = df["Spending"].rank(method="average", pct=True)

    try:
        df["CabinNumQuantile"] = pd.qcut(
            df["CabinNum"], q=10, labels=False, duplicates="drop"
        )
    except ValueError:
        df["CabinNumQuantile"] = np.nan

    try:
        df["SpendingQuantile"] = pd.qcut(
            df["Spending"], q=10, labels=False, duplicates="drop"
        )
    except ValueError:
        df["SpendingQuantile"] = np.nan

    df = df.drop(columns=["Cabin", "Name", "PassengerId"])
    return df


y = train["Transported"].astype(int)
X = engineer_features(train.drop(columns=["Transported"]))
X_test = engineer_features(test.copy())

cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ],
    remainder="drop",
)

base_lgbm_params = dict(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=24,
    max_depth=-1,
    min_child_samples=25,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.0,
    objective="binary",
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(
    [
        ("prep", preprocess),
        ("clf", LGBMClassifier(**base_lgbm_params)),
    ]
)

seed_models = []
for seed in [42, 2023, 7]:
    seed_model = Pipeline(
        [
            ("prep", preprocess),
            ("clf", LGBMClassifier(**{**base_lgbm_params, "random_state": seed})),
        ]
    )
    seed_models.append(seed_model)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
valid_acc = accuracy_score(y_valid, valid_pred)

model.fit(X, y)
test_pred = model.predict(X_test).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {valid_acc:.6f}")
