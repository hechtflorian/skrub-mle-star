
import os
import sys
import subprocess
import random
import warnings

warnings.filterwarnings("ignore")


def ensure_package(import_name, pip_name=None):
    if pip_name is None:
        pip_name = import_name
    package_spec = pip_name
    if import_name == "sklearn":
        package_spec = "scikit-learn"
    if import_name == "catboost":
        package_spec = "catboost"
    if import_name == "xgboost":
        package_spec = "xgboost"
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_spec],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if installed.returncode != 0:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])


ensure_package("numpy")
ensure_package("pandas")
ensure_package("sklearn", "scikit-learn")
ensure_package("catboost")
ensure_package("xgboost")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = 'NObeyesdad'
ID = 'id'

train_path = './input/train.csv'
test_path = './input/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

cat_cols = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'
]

feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
X = train[feature_cols].copy()
y = train[TARGET].copy()
X_test = test[feature_cols].copy()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_xgb = [c for c in feature_cols if c not in num_cols]

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

X_train, X_valid, y_train, y_valid, y_train_enc, y_valid_enc = train_test_split(
    X,
    y,
    y_enc,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

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
            cat_cols_xgb
        )
    ],
    remainder="drop"
)

X_train_t = preprocessor.fit_transform(X_train)
X_valid_t = preprocessor.transform(X_valid)

cat_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=SEED,
    verbose=200
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

xgb_model = XGBClassifier(
    objective="multi:softprob",
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

xgb_model.fit(X_train_t, y_train_enc)

cat_valid_proba = cat_model.predict_proba(X_valid)
xgb_valid_proba = xgb_model.predict_proba(X_valid_t)

cat_valid_proba = np.asarray(cat_valid_proba)
xgb_valid_proba = np.asarray(xgb_valid_proba)

ensemble_valid_proba = 0.5 * cat_valid_proba + 0.5 * xgb_valid_proba
ensemble_valid_pred_idx = np.argmax(ensemble_valid_proba, axis=1)
valid_pred = label_encoder.inverse_transform(ensemble_valid_pred_idx)

final_validation_score = accuracy_score(y_valid, valid_pred)
print(f'Final Validation Performance: {final_validation_score}')

final_cat_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=SEED,
    verbose=200
)

final_cat_model.fit(
    X,
    y,
    cat_features=cat_cols
)

X_full_t = preprocessor.fit_transform(X)
X_test_t = preprocessor.transform(X_test)

final_xgb_model = XGBClassifier(
    objective="multi:softprob",
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

final_xgb_model.fit(X_full_t, y_enc)

cat_test_proba = np.asarray(final_cat_model.predict_proba(X_test))
xgb_test_proba = np.asarray(final_xgb_model.predict_proba(X_test_t))

ensemble_test_proba = 0.5 * cat_test_proba + 0.5 * xgb_test_proba
test_pred_idx = np.argmax(ensemble_test_proba, axis=1)
test_pred = label_encoder.inverse_transform(test_pred_idx)

submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred
})
submission.to_csv('submission.csv', index=False)
