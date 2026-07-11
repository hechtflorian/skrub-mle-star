
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_map = {"Introvert": 0, "Extrovert": 1}
inv_target_map = {0: "Introvert", 1: "Extrovert"}

y = train["Personality"].map(target_map).astype(int)
X = train.drop(columns=["Personality"]).copy()
X_test = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == "object"]

for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]
for col in num_cols:
    med = X[col].median()
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

X["id"] = X["id"].fillna(-1)
X_test["id"] = X_test["id"].fillna(-1)

X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=SEED,
    subsample=0.9,
    colsample_bytree=0.9
)

model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc}")

test_pred = model.predict(X_test)
submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pd.Series(test_pred).map(inv_target_map)
})
submission.to_csv("submission.csv", index=False)
