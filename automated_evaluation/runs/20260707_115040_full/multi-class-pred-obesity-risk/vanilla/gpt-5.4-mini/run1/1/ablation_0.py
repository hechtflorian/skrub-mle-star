
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Paths
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")
submission_path = "submission.csv"

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "NObeyesdad"
y = train[target].copy()
X = train.drop(columns=[target]).copy()

# Keep id for submission
test_ids = test["id"].copy()

# Drop id from features
if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in test.columns:
    X_test = test.drop(columns=["id"]).copy()
else:
    X_test = test.copy()

# Detect categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

def prepare_data_for_catboost(df, use_cat_features=True):
    df = df.copy()
    if not use_cat_features:
        for col in cat_cols:
            df[col] = df[col].astype("category").cat.codes.astype(np.int32)
    return df

def train_and_score(name, params=None, use_cat_features=True, use_best_model=True):
    params = params or {}

    X_tr_p = prepare_data_for_catboost(X_tr, use_cat_features=use_cat_features)
    X_val_p = prepare_data_for_catboost(X_val, use_cat_features=use_cat_features)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=500,           # reduced to avoid timeout
        depth=6,                  # slightly smaller to speed up
        learning_rate=0.08,
        random_seed=42,
        verbose=False,
        thread_count=-1,
        **params
    )

    fit_kwargs = {}
    if use_cat_features:
        fit_kwargs["cat_features"] = cat_cols
    if use_best_model:
        fit_kwargs["eval_set"] = (X_val_p, y_val)
        fit_kwargs["use_best_model"] = True

    model.fit(X_tr_p, y_tr, **fit_kwargs)
    val_pred = model.predict(X_val_p).ravel()
    score = accuracy_score(y_val, val_pred)
    print(f"{name}: validation accuracy = {score:.6f}")
    return score, model

baseline_score, baseline_model = train_and_score(
    "Baseline", use_cat_features=True, use_best_model=True
)

ablation_scores = {}

ablation_scores["No cat_features"], _ = train_and_score(
    "Ablation - No cat_features",
    use_cat_features=False,
    use_best_model=True
)

ablation_scores["No best_model"], _ = train_and_score(
    "Ablation - No best_model",
    use_cat_features=True,
    use_best_model=False
)

print("\nPerformance impact relative to baseline:")
for name, score in ablation_scores.items():
    diff = score - baseline_score
    print(f"{name}: {diff:+.6f}")

drops = {name: baseline_score - score for name, score in ablation_scores.items()}
most_important = max(drops, key=drops.get)

print("\nMost important part of the original code:")
print(f"{most_important} (largest accuracy drop when removed: {drops[most_important]:.6f})")

# Final validation performance line for parsing
final_validation_score = baseline_score
print(f"Final Validation Performance: {final_validation_score}")

# Fit final model on full training data and predict test set
final_model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=500,      # reduced to avoid timeout
    depth=6,
    learning_rate=0.08,
    random_seed=42,
    verbose=False,
    thread_count=-1
)

final_model.fit(X, y_encoded, cat_features=cat_cols)

test_pred_encoded = final_model.predict(X_test).ravel()
test_pred = label_encoder.inverse_transform(test_pred_encoded.astype(int))

submission = pd.DataFrame({
    "id": test_ids,
    "NObeyesdad": test_pred
})
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
