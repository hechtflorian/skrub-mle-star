
import os
import random
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.isotonic import IsotonicRegression
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def feature_engineering(df):
    df = df.copy()

    # Cabin split
    cabin_split = df["Cabin"].fillna("NA/NA/NA").astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0].astype(str)
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2].astype(str)

    # Passenger group features
    df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0].astype(str)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count").astype(int)

    # Spending features
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spending_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

    # Cheap high-signal interactions
    df["AnySpending"] = (df[spending_cols].fillna(0).sum(axis=1) > 0).astype(int)
    df["SpendingPerPerson"] = df["TotalSpending"] / df["GroupSize"].replace(0, np.nan)

    # Optional per-category nonzero indicators
    for col in spending_cols:
        df[f"{col}_NonZero"] = (df[col].fillna(0) > 0).astype(int)

    # Missingness indicators
    for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]:
        if col in df.columns:
            df[col + "_isna"] = df[col].isna().astype(int)

    # Keep raw cabin/group signals and core categoricals for CatBoost
    categorical_like = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group"]
    for c in categorical_like:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    # Keep booleans as strings to avoid CatBoost categorical type issues
    for c in ["CryoSleep", "VIP"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA").astype(str)

    return df


train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

y = train_fe["Transported"].astype(int)
X = train_fe.drop(columns=["Transported"])

# Align test to train columns
X_test = test_fe.reindex(columns=X.columns, fill_value=np.nan).copy()

# Identify categorical columns explicitly and ensure they are string-like
cat_cols = [
    c for c in X.columns
    if X[c].dtype == "object" or str(X[c].dtype).startswith("string")
]

# Convert categorical columns in both train/test to string with a single missing token
for c in cat_cols:
    X[c] = X[c].astype("string").fillna("NA").astype(str)
    X_test[c] = X_test[c].astype("string").fillna("NA").astype(str)

# Fill numeric missing values
num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    med = X[c].median()
    if pd.isna(med):
        med = 0
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

# Hold-out validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Model 1: original CatBoost
model1 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)

model1.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# Model 2: same family, slightly different configuration to encourage complementarity
model2 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=2500,
    depth=8,
    learning_rate=0.025,
    random_seed=SEED + 1,
    verbose=0,
    eval_metric="Accuracy",
    l2_leaf_reg=4.0,
    subsample=0.85,
    rsm=0.85
)

model2.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# Validation probabilities
p1_val = model1.predict_proba(X_val)[:, 1]
p2_val = model2.predict_proba(X_val)[:, 1]

# Tiny calibration step: clip + isotonic calibration on validation split
eps = 1e-6
p1_val = np.clip(p1_val, eps, 1 - eps)
p2_val = np.clip(p2_val, eps, 1 - eps)

cal1 = IsotonicRegression(out_of_bounds="clip")
cal2 = IsotonicRegression(out_of_bounds="clip")
cal1.fit(p1_val, y_val)
cal2.fit(p2_val, y_val)

p1_val_cal = np.clip(cal1.transform(p1_val), eps, 1 - eps)
p2_val_cal = np.clip(cal2.transform(p2_val), eps, 1 - eps)

pred1_val = (p1_val_cal >= 0.5).astype(int)
pred2_val = (p2_val_cal >= 0.5).astype(int)

acc1 = accuracy_score(y_val, pred1_val)
acc2 = accuracy_score(y_val, pred2_val)

better_is_model1 = acc1 >= acc2

# Learn gate thresholds from validation split once
d_val = np.abs(p1_val_cal - p2_val_cal)

# Candidate thresholds over quantiles for a simple gated ensemble
q1 = float(np.quantile(d_val, 0.33))
q2 = float(np.quantile(d_val, 0.66))
if q1 >= q2:
    q1, q2 = 0.15, 0.35

def gated_blend(p1, p2, better_model1=True, t1=q1, t2=q2):
    d = np.abs(p1 - p2)
    if better_model1:
        better = p1
        other = p2
    else:
        better = p2
        other = p1

    blended = np.empty_like(p1, dtype=float)

    small = d <= t1
    medium = (d > t1) & (d <= t2)
    large = d > t2

    blended[small] = 0.5 * p1[small] + 0.5 * p2[small]
    blended[medium] = 0.65 * better[medium] + 0.35 * other[medium]
    blended[large] = better[large]

    return np.clip(blended, 1e-6, 1 - 1e-6)

# Validation ensemble performance
val_blend = gated_blend(p1_val_cal, p2_val_cal, better_model1=better_is_model1)
val_pred = (val_blend >= 0.5).astype(int)
val_acc = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {val_acc:.6f}")

# Train on full data and predict test set
final_model1 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=model1.get_best_iteration() if model1.get_best_iteration() is not None else 2000,
    depth=6,
    learning_rate=0.03,
    random_seed=SEED,
    verbose=0,
    eval_metric="Accuracy"
)

final_model2 = CatBoostClassifier(
    loss_function="Logloss",
    iterations=model2.get_best_iteration() if model2.get_best_iteration() is not None else 2500,
    depth=8,
    learning_rate=0.025,
    random_seed=SEED + 1,
    verbose=0,
    eval_metric="Accuracy",
    l2_leaf_reg=4.0,
    subsample=0.85,
    rsm=0.85
)

final_model1.fit(X, y, cat_features=cat_cols)
final_model2.fit(X, y, cat_features=cat_cols)

p1_test = final_model1.predict_proba(X_test)[:, 1]
p2_test = final_model2.predict_proba(X_test)[:, 1]

p1_test = np.clip(cal1.transform(np.clip(p1_test, eps, 1 - eps)), eps, 1 - eps)
p2_test = np.clip(cal2.transform(np.clip(p2_test, eps, 1 - eps)), eps, 1 - eps)

test_blend = gated_blend(p1_test, p2_test, better_model1=better_is_model1)
test_pred = (test_blend >= 0.5).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)
