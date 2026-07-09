
import os
import random
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"
os.makedirs("./final", exist_ok=True)

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

model1 = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    random_state=SEED,
    subsample=0.9,
    colsample_bytree=0.9,
)

model1.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
)

oof_1 = model1.predict_proba(X_val)[:, 1]
test_1 = model1.predict_proba(T_enc)[:, 1]

model2 = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.02,
    num_leaves=48,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=SEED + 7,
)

model2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
)

oof_2 = model2.predict_proba(X_val)[:, 1]
test_2 = model2.predict_proba(T_enc)[:, 1]

def percentile_rank(x):
    x = np.asarray(x)
    return rankdata(x, method="average") / len(x)

r_oof_1 = percentile_rank(oof_1)
r_oof_2 = percentile_rank(oof_2)
r_test_1 = percentile_rank(test_1)
r_test_2 = percentile_rank(test_2)

best_w = 0.5
best_auc = -np.inf

for w in np.arange(0.35, 0.651, 0.01):
    blend_oof = w * r_oof_1 + (1.0 - w) * r_oof_2
    auc = roc_auc_score(y_val, blend_oof)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)

agreement = np.abs(r_oof_1 - r_oof_2)
blend_oof = best_w * r_oof_1 + (1.0 - best_w) * r_oof_2
final_oof = np.where(agreement > 0.35, 0.9 * blend_oof + 0.1 * 0.5, blend_oof)

val_pred = final_oof
final_validation_score = roc_auc_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_blend = best_w * r_test_1 + (1.0 - best_w) * r_test_2
test_agreement = np.abs(r_test_1 - r_test_2)
final_test = np.where(test_agreement > 0.35, 0.9 * test_blend + 0.1 * 0.5, test_blend)

id_col = "EmployeeNumber" if "EmployeeNumber" in test.columns else "id"
submission = pd.DataFrame({
    "EmployeeNumber": test[id_col],
    "Attrition": final_test
})
submission.to_csv("./final/submission.csv", index=False)
