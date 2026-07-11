
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool

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

# Identify column types
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]

# Basic preprocessing helper
def preprocess(df, cat_cols, num_cols, num_fill_values=None):
    df = df.copy()
    if num_fill_values is None:
        num_fill_values = {}
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
    for col in num_cols:
        fill_value = num_fill_values.get(col, df[col].median())
        df[col] = df[col].fillna(fill_value)
    df["id"] = df["id"].fillna(-1)
    return df

# Compute train medians for numeric columns
num_fill_values = {col: X[col].median() for col in num_cols}

X = preprocess(X, cat_cols, num_cols, num_fill_values=num_fill_values)
X_test = preprocess(X_test, cat_cols, num_cols, num_fill_values=num_fill_values)

# Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

def fit_eval(X_tr, y_tr, X_val, y_val, cat_features, iterations=1500, depth=6, lr=0.03, use_best_model=True):
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=SEED,
        verbose=0
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=use_best_model)
    pred = model.predict(X_val).astype(int).ravel()
    return accuracy_score(y_val, pred), model

baseline_acc, baseline_model = fit_eval(
    X_tr, y_tr, X_val, y_val, cat_cols,
    iterations=1500, depth=6, lr=0.03, use_best_model=True
)
print(f"Baseline validation accuracy: {baseline_acc:.6f}")

# Ablation 1: properly disable categorical features by encoding them numerically
# Here, we use simple ordinal/category codes to avoid passing strings as numeric.
def encode_categoricals_as_codes(train_df, val_df, test_df, cat_cols):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        combined = pd.concat(
            [train_df[col], val_df[col], test_df[col]],
            axis=0
        ).astype(str).fillna("missing")
        categories = pd.Index(combined.unique())

        train_df[col] = pd.Categorical(train_df[col].astype(str).fillna("missing"), categories=categories).codes
        val_df[col] = pd.Categorical(val_df[col].astype(str).fillna("missing"), categories=categories).codes
        test_df[col] = pd.Categorical(test_df[col].astype(str).fillna("missing"), categories=categories).codes

    return train_df, val_df, test_df

X_tr_no_cat, X_val_no_cat, X_test_no_cat = encode_categoricals_as_codes(
    X_tr, X_val, X_test, cat_cols
)

# No categorical features passed now; all columns are numeric
ablation_no_cat_acc, _ = fit_eval(
    X_tr_no_cat, y_tr, X_val_no_cat, y_val,
    cat_features=[],
    iterations=1500, depth=6, lr=0.03, use_best_model=True
)
print(f"Ablation 1 (disable categorical features) accuracy: {ablation_no_cat_acc:.6f} | delta: {ablation_no_cat_acc - baseline_acc:+.6f}")

# Ablation 2: remove best-model early stopping behavior
ablation_no_best_acc, _ = fit_eval(
    X_tr, y_tr, X_val, y_val,
    cat_features=cat_cols,
    iterations=1500, depth=6, lr=0.03, use_best_model=False
)
print(f"Ablation 2 (disable use_best_model) accuracy: {ablation_no_best_acc:.6f} | delta: {ablation_no_best_acc - baseline_acc:+.6f}")

# Optional extra simple ablation: reduce model capacity
smaller_model_acc, _ = fit_eval(
    X_tr, y_tr, X_val, y_val,
    cat_features=cat_cols,
    iterations=1500, depth=4, lr=0.03, use_best_model=True
)
print(f"Ablation 3 (reduce depth to 4) accuracy: {smaller_model_acc:.6f} | delta: {smaller_model_acc - baseline_acc:+.6f}")

results = {
    "baseline": baseline_acc,
    "disable_categorical_features": ablation_no_cat_acc,
    "disable_use_best_model": ablation_no_best_acc,
    "reduce_depth": smaller_model_acc,
}

best_ablation = max(
    [k for k in results.keys() if k != "baseline"],
    key=lambda k: results[k]
)

largest_gain = results[best_ablation] - results["baseline"]
print(f"Most important part for performance: {best_ablation} (largest accuracy gain: {largest_gain:+.6f})")

# Final training on full data using best practical setup
final_model = CatBoostClassifier(
    iterations=1500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=SEED,
    verbose=0
)
full_pool = Pool(X, y, cat_features=cat_cols)
final_model.fit(full_pool)

# Validation performance line required by prompt
final_validation_score = baseline_acc
print(f"Final Validation Performance: {final_validation_score}")

# Predict test set
test_pool = Pool(X_test, cat_features=cat_cols)
test_pred = final_model.predict(test_pool).astype(int).ravel()
test_labels = pd.Series(test_pred).map(inv_target_map)

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": test_labels
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
