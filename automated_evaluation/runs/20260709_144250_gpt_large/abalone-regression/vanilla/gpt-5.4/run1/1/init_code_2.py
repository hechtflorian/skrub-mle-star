
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

target_col = "Rings"
features = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Whole weight.1",
    "Whole weight.2",
    "Shell weight",
]
cat_cols = ["Sex"]
num_cols = [c for c in features if c not in cat_cols]

missing_train = [c for c in features + [target_col] if c not in train.columns]
missing_test = [c for c in features if c not in test.columns]
if missing_train:
    raise ValueError(f"Missing columns in train.csv: {missing_train}")
if missing_test:
    raise ValueError(f"Missing columns in test.csv: {missing_test}")

df = train[features + [target_col]].copy()
test_df = test[["id"] + features].copy()

df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).clip(lower=0)

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

for c in cat_cols:
    df[c] = df[c].fillna("Unknown").astype(str)
    test_df[c] = test_df[c].fillna("Unknown").astype(str)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

base_model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.03,
    max_iter=600,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=SEED,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", base_model),
    ]
)

y_train_log = np.log1p(train_df[target_col].values)
y_val = val_df[target_col].values

model.fit(train_df[features], y_train_log)

val_pred_log = model.predict(val_df[features])
val_pred = np.expm1(val_pred_log)
val_pred = np.clip(val_pred, 0, None)

final_validation_score = np.sqrt(mean_squared_log_error(y_val, val_pred))

full_model = clone(model)
y_full_log = np.log1p(df[target_col].values)
full_model.fit(df[features], y_full_log)

test_pred_log = full_model.predict(test_df[features])
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "Rings": test_pred,
    }
)
submission.to_csv("submission_fttransformer.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
