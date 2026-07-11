
import os
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = train["Attrition"].astype(int)
X = train.drop(columns=["Attrition"]).copy()
T = test.copy()

for col in X.columns:
    if col in T.columns:
        if X[col].dtype == "object" or T[col].dtype == "object":
            X[col] = X[col].astype("object").fillna("NA").astype(str)
            T[col] = T[col].astype("object").fillna("NA").astype(str)
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            T[col] = T[col].fillna(median_val)

all_df = pd.concat([X, T], axis=0, ignore_index=True)
all_df = pd.get_dummies(all_df, dummy_na=True)

X_enc = all_df.iloc[:len(X)].copy()
T_enc = all_df.iloc[len(X):].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X_enc, y, test_size=0.2, random_state=SEED, stratify=y
)

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    random_state=SEED,
    subsample=0.9,
    colsample_bytree=0.9,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[
        early_stopping(stopping_rounds=50, verbose=False),
    ],
)

val_pred = model.predict_proba(X_val)[:, 1]
final_validation_score = roc_auc_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = model.predict_proba(T_enc)[:, 1]

id_col = "EmployeeNumber" if "EmployeeNumber" in test.columns else "id"
submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": test_pred
})
submission.to_csv("submission.csv", index=False)
