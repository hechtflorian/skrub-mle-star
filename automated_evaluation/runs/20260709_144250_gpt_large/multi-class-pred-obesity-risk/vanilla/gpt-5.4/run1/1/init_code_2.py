
import os
import sys
import subprocess
import random
import warnings

warnings.filterwarnings("ignore")


def ensure_package(import_name, pip_name=None):
    if pip_name is None:
        pip_name = import_name
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_package("numpy")
ensure_package("pandas")
ensure_package("sklearn", "scikit-learn")
ensure_package("xgboost")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = "NObeyesdad"
ID = "id"

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
X = train[feature_cols].copy()
y = train[TARGET].copy()
X_test = test[feature_cols].copy()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]),
            cat_cols
        )
    ],
    remainder="drop"
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=SEED,
    stratify=y_enc
)

X_train_t = preprocessor.fit_transform(X_train)
X_valid_t = preprocessor.transform(X_valid)

model = XGBClassifier(
    objective="multi:softmax",
    num_class=num_classes,
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=-1
)

model.fit(X_train_t, y_train)

valid_pred = model.predict(X_valid_t)
valid_pred = np.asarray(valid_pred).astype(int).reshape(-1)
final_validation_score = accuracy_score(y_valid, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

X_full_t = preprocessor.fit_transform(X)
X_test_t = preprocessor.transform(X_test)

final_model = XGBClassifier(
    objective="multi:softmax",
    num_class=num_classes,
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=-1
)

final_model.fit(X_full_t, y_enc)

test_pred = final_model.predict(X_test_t)
test_pred = np.asarray(test_pred).astype(int).reshape(-1)
test_pred_labels = label_encoder.inverse_transform(test_pred)

submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred_labels
})
submission.to_csv("submission.csv", index=False)
print(submission.head())
