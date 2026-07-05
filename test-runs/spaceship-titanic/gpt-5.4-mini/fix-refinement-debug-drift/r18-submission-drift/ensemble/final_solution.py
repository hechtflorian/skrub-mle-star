
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

input_dir = "./input"
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

target_col = "Transported"
random_state = 42

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def add_features(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        else:
            df["CabinDeck"] = np.nan
            df["CabinNum"] = np.nan
            df["CabinSide"] = np.nan
    if "Name" in df.columns:
        df["NameLength"] = df["Name"].astype(str).str.len()
    if "PassengerId" in df.columns:
        df["GroupSize"] = df["PassengerId"].astype(str).str.split("_").str[0].map(
            df["PassengerId"].astype(str).str.split("_").str[0].value_counts()
        )
    total_spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_spend_cols = [c for c in total_spend_cols if c in df.columns]
    if existing_spend_cols:
        df["TotalSpend"] = df[existing_spend_cols].fillna(0).sum(axis=1)
        df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    return df

train_part_fe = add_features(train_part)
valid_part_fe = add_features(valid_part)

X_train = train_part_fe.drop(columns=[target_col], errors="ignore")
y_train = train_part_fe[target_col].astype(int)

X_valid = valid_part_fe.drop(columns=[target_col], errors="ignore")
y_valid = valid_part_fe[target_col].astype(int)

numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=random_state,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

pipeline.fit(X_train, y_train)
valid_pred = pipeline.predict(X_valid)
final_validation_score = accuracy_score(y_valid, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Smallest fix for the reported export-contract error: write the required submission file.
os.makedirs("./final", exist_ok=True)

# Fit on full training data for submission export
train_df_fe = add_features(train_df)
test_df_fe = add_features(test_df)

X_full = train_df_fe.drop(columns=[target_col], errors="ignore")
y_full = train_df_fe[target_col].astype(int)
X_test = test_df_fe.copy()

pipeline.fit(X_full, y_full)
test_pred = pipeline.predict(X_test)
submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred.astype(bool),
    }
)
submission.to_csv("./final/submission.csv", index=False)
