import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
train = pd.read_csv(train_path)

y = train["Attrition"].astype(int)
X = train.drop(columns=["Attrition"]).copy()

# Base preprocessing: fill missing values + one-hot encoding
X_base = X.copy()

for col in X_base.columns:
    if X_base[col].dtype == "object":
        X_base[col] = X_base[col].astype(str).fillna("NA")
    else:
        X_base[col] = X_base[col].fillna(X_base[col].median())

X_base = pd.get_dummies(X_base, dummy_na=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_base, y, test_size=0.2, random_state=SEED, stratify=y
)

def train_eval(X_tr, X_va, y_tr, y_va, desc):
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=SEED,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
    )
    pred = model.predict_proba(X_va)[:, 1]
    score = roc_auc_score(y_va, pred)
    print(f"{desc}: AUC = {score:.6f}")
    return score

# Baseline
baseline_auc = train_eval(X_train, X_val, y_train, y_val, "Baseline")

# Ablation 1: Disable one-hot encoding by using only numeric columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns
X_num = X[numeric_cols].copy()
for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())

X_train_num, X_val_num, y_train_num, y_val_num = train_test_split(
    X_num, y, test_size=0.2, random_state=SEED, stratify=y
)
num_auc = train_eval(X_train_num, X_val_num, y_train_num, y_val_num, "Ablation: no categorical one-hot encoding")

# Ablation 2: Disable target validation-driven early fitting effect by reducing tree count
# (same features, fewer estimators)
def train_eval_fewer_trees(X_tr, X_va, y_tr, y_va, desc):
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=31,
        random_state=SEED,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
    )
    pred = model.predict_proba(X_va)[:, 1]
    score = roc_auc_score(y_va, pred)
    print(f"{desc}: AUC = {score:.6f}")
    return score

fewer_trees_auc = train_eval_fewer_trees(X_train, X_val, y_train, y_val, "Ablation: fewer estimators")

print("\nAblation impact vs baseline:")
print(f"No categorical one-hot encoding: {num_auc - baseline_auc:+.6f}")
print(f"Fewer estimators: {fewer_trees_auc - baseline_auc:+.6f}")

impacts = {
    "one-hot encoding": baseline_auc - num_auc,
    "more boosting rounds": baseline_auc - fewer_trees_auc,
}

most_important = max(impacts, key=impacts.get)
print(f"\nMost important part contributing to overall performance: {most_important}")