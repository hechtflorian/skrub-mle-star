

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "NObeyesdad"
y = train[target]
X = train.drop(columns=[target])

# Manual categorical encoding instead of CatBoost native cat_features
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit encoder on training split only
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_tr_enc = X_tr.copy()
X_val_enc = X_val.copy()
test_enc = test.copy()

if cat_cols:
    X_tr_enc[cat_cols] = encoder.fit_transform(X_tr[cat_cols].astype(str))
    X_val_enc[cat_cols] = encoder.transform(X_val[cat_cols].astype(str))
    test_enc[cat_cols] = encoder.transform(test[cat_cols].astype(str))

# Ensure numeric columns are preserved and aligned
X_tr_enc = X_tr_enc.drop(columns=["id"], errors="ignore")
X_val_enc = X_val_enc.drop(columns=["id"], errors="ignore")
test_enc = test_enc.drop(columns=["id"], errors="ignore")

# Try a small fixed set of stronger regularization settings
candidate_params = [
    {"depth": 6, "min_data_in_leaf": 30, "l2_leaf_reg": 8},
    {"depth": 7, "min_data_in_leaf": 50, "l2_leaf_reg": 10},
    {"depth": 8, "min_data_in_leaf": 20, "l2_leaf_reg": 12},
]

best_score = -np.inf
best_model = None
best_params = None

for params in candidate_params:
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2000,
        learning_rate=0.05,
        random_seed=42,
        verbose=200,
        use_best_model=True,
        **params
    )
    model.fit(
        X_tr_enc,
        y_tr,
        eval_set=(X_val_enc, y_val),
        use_best_model=True
    )
    val_pred = model.predict(X_val_enc).ravel()
    score = accuracy_score(y_val, val_pred)
    print(f"Params={params}, Validation Accuracy={score}")

    if score > best_score:
        best_score = score
        best_model = model
        best_params = params

print(f"Best Params: {best_params}")
print(f"Final Validation Performance: {best_score}")

test_pred = best_model.predict(test_enc).ravel()
submission = pd.DataFrame({
    "id": test["id"],
    "NObeyesdad": test_pred
})
submission.to_csv("submission.csv", index=False)
