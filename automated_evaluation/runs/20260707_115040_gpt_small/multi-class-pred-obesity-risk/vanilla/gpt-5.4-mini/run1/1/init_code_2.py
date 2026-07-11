
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "NObeyesdad"
y = train[target]
X = train.drop(columns=[target])

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Keep a copy of ids for submission before any transformations
test_ids = test["id"].copy()

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Ordinal encode categorical columns so LightGBM receives only numeric data
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_tr = X_tr.copy()
X_val = X_val.copy()
test_enc = test.copy()

X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
X_val[cat_cols] = enc.transform(X_val[cat_cols])
test_enc[cat_cols] = enc.transform(test_enc[cat_cols])

# Drop id from features
X_tr = X_tr.drop(columns=["id"])
X_val = X_val.drop(columns=["id"])
test_features = test_enc.drop(columns=["id"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = LGBMClassifier(
    objective="multiclass",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    random_state=42,
    class_weight=None
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss"
)

val_pred = model.predict(X_val)
final_validation_score = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = model.predict(test_features)
submission = pd.DataFrame({
    "id": test_ids,
    "NObeyesdad": test_pred
})
submission.to_csv("submission.csv", index=False)
