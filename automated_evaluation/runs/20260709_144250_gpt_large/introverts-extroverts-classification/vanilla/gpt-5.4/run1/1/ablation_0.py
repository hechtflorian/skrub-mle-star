
import os
import sys
import subprocess
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

train_path = os.path.join(".", "input", "train.csv")
test_path = os.path.join(".", "input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = (train["Personality"] == "Extrovert").astype(int)
X_raw = train.drop(columns=["Personality"]).copy()

X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
    X_raw,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

cat_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

def preprocess(train_df, valid_df, fill_cat=True, fill_num=True):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    # CatBoost requires categorical columns to contain only strings or integers.
    # Even when we do not want to "fill" categorical missing values for the ablation,
    # we still need to convert NaN to a valid string token to avoid runtime failure.
    for c in cat_cols:
        if fill_cat:
            train_df[c] = train_df[c].fillna("Missing")
            valid_df[c] = valid_df[c].fillna("Missing")
        else:
            train_df[c] = train_df[c].where(~train_df[c].isna(), "__NaN__")
            valid_df[c] = valid_df[c].where(~valid_df[c].isna(), "__NaN__")
        train_df[c] = train_df[c].astype(str)
        valid_df[c] = valid_df[c].astype(str)

    if fill_num:
        for c in num_cols:
            median_value = train_df[c].median()
            train_df[c] = train_df[c].fillna(median_value)
            valid_df[c] = valid_df[c].fillna(median_value)

    return train_df, valid_df

def evaluate_ablation(name, fill_cat=True, fill_num=True, use_best_model=True):
    X_train, X_valid = preprocess(
        X_train_raw,
        X_valid_raw,
        fill_cat=fill_cat,
        fill_num=fill_num
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Accuracy",
        verbose=0,
        random_state=RANDOM_STATE
    )

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "cat_features": cat_cols,
        "eval_set": (X_valid, y_valid)
    }

    if use_best_model:
        fit_kwargs["use_best_model"] = True

    model.fit(**fit_kwargs)

    pred = model.predict(X_valid)
    pred = np.array(pred).astype(int).ravel()
    score = accuracy_score(y_valid, pred)
    print(f"{name}: accuracy={score:.6f}")
    return score

results = {}

results["baseline"] = evaluate_ablation(
    "baseline",
    fill_cat=True,
    fill_num=True,
    use_best_model=True
)

results["no_categorical_missing_fill"] = evaluate_ablation(
    "no_categorical_missing_fill",
    fill_cat=False,
    fill_num=True,
    use_best_model=True
)

results["no_numeric_missing_fill"] = evaluate_ablation(
    "no_numeric_missing_fill",
    fill_cat=True,
    fill_num=False,
    use_best_model=True
)

results["no_best_model_selection"] = evaluate_ablation(
    "no_best_model_selection",
    fill_cat=True,
    fill_num=True,
    use_best_model=False
)

baseline_score = results["baseline"]
drops = {k: baseline_score - v for k, v in results.items() if k != "baseline"}
most_important_part = max(drops, key=drops.get)

print(f"baseline_accuracy={baseline_score:.6f}")
for name, drop in drops.items():
    print(f"impact_of_{name}={drop:+.6f}")
print(f"largest_contributor={most_important_part}")

final_validation_score = baseline_score
print(f"Final Validation Performance: {final_validation_score}")

# Train final model on full training data using baseline preprocessing
def preprocess_full(train_df, test_df, fill_cat=True, fill_num=True):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for c in cat_cols:
        if fill_cat:
            train_df[c] = train_df[c].fillna("Missing")
            test_df[c] = test_df[c].fillna("Missing")
        else:
            train_df[c] = train_df[c].where(~train_df[c].isna(), "__NaN__")
            test_df[c] = test_df[c].where(~test_df[c].isna(), "__NaN__")
        train_df[c] = train_df[c].astype(str)
        test_df[c] = test_df[c].astype(str)

    if fill_num:
        for c in num_cols:
            median_value = train_df[c].median()
            train_df[c] = train_df[c].fillna(median_value)
            test_df[c] = test_df[c].fillna(median_value)

    return train_df, test_df

X_full = train.drop(columns=["Personality"]).copy()
X_test = test.copy()

X_full_proc, X_test_proc = preprocess_full(
    X_full,
    X_test,
    fill_cat=True,
    fill_num=True
)

final_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=0,
    random_state=RANDOM_STATE
)

final_model.fit(X_full_proc, y, cat_features=cat_cols)

test_pred = final_model.predict(X_test_proc)
test_pred = np.array(test_pred).astype(int).ravel()
test_labels = np.where(test_pred == 1, "Extrovert", "Introvert")

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": test_labels
})

submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
