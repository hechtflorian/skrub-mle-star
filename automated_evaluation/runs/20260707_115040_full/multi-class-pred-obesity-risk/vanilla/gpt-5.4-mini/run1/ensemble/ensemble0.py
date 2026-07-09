
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

# Preserve original ordered label list for consistent class alignment
class_labels = list(train[target].unique())
class_to_idx = {c: i for i, c in enumerate(class_labels)}

# Use the same train/validation split for both base models
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

########################################
# Solution 1: Ordinal encoding + CatBoost
########################################
cat_cols_1 = X.select_dtypes(include=["object"]).columns.tolist()

encoder_1 = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_tr_enc = X_tr.copy()
X_val_enc = X_val.copy()
test_enc = test.copy()

if cat_cols_1:
    X_tr_enc[cat_cols_1] = encoder_1.fit_transform(X_tr[cat_cols_1].astype(str))
    X_val_enc[cat_cols_1] = encoder_1.transform(X_val[cat_cols_1].astype(str))
    test_enc[cat_cols_1] = encoder_1.transform(test[cat_cols_1].astype(str))

X_tr_enc = X_tr_enc.drop(columns=["id"], errors="ignore")
X_val_enc = X_val_enc.drop(columns=["id"], errors="ignore")
test_enc = test_enc.drop(columns=["id"], errors="ignore")

candidate_params = [
    {"depth": 6, "min_data_in_leaf": 30, "l2_leaf_reg": 8},
    {"depth": 7, "min_data_in_leaf": 50, "l2_leaf_reg": 10},
    {"depth": 8, "min_data_in_leaf": 20, "l2_leaf_reg": 12},
]

best_score_1 = -np.inf
best_model_1 = None
best_params_1 = None

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
    print(f"Solution 1 Params={params}, Validation Accuracy={score}")

    if score > best_score_1:
        best_score_1 = score
        best_model_1 = model
        best_params_1 = params

print(f"Best Solution 1 Params: {best_params_1}")
print(f"Solution 1 Validation Performance: {best_score_1}")

val_pred_1 = best_model_1.predict(X_val_enc).ravel()
test_proba_1_raw = best_model_1.predict_proba(test_enc)

# Align Solution 1 probabilities to the common ordered class list
model_classes_1 = list(best_model_1.classes_)
proba_1 = np.zeros((test_proba_1_raw.shape[0], len(class_labels)), dtype=np.float64)
for i, cls in enumerate(model_classes_1):
    if cls in class_to_idx:
        proba_1[:, class_to_idx[cls]] = test_proba_1_raw[:, i]

val_proba_1_raw = best_model_1.predict_proba(X_val_enc)
val_proba_1 = np.zeros((val_proba_1_raw.shape[0], len(class_labels)), dtype=np.float64)
for i, cls in enumerate(model_classes_1):
    if cls in class_to_idx:
        val_proba_1[:, class_to_idx[cls]] = val_proba_1_raw[:, i]

########################################
# Solution 2: Leakage-safe target encoding + seed averaging
########################################
# Implement a simple, robust target encoding solution without changing the overall logic much
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

cat_cols_2 = X.select_dtypes(include=["object"]).columns.tolist()
num_cols_2 = [c for c in X.columns if c not in cat_cols_2 and c != "id"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# We will map probabilities back to original class labels using label_encoder.classes_
# and then align them to class_labels
n_classes = len(label_encoder.classes_)

def target_encode_train_valid_test(X_train_df, y_train_enc, X_valid_df, X_test_df, cols, n_splits=5, smoothing=20, seed=42):
    X_train_enc = X_train_df.copy()
    X_valid_enc = X_valid_df.copy()
    X_test_enc = X_test_df.copy()

    global_mean = np.mean(y_train_enc) if len(np.unique(y_train_enc)) > 1 else 0.0
    for col in cols:
        oof_map = pd.DataFrame(index=X_train_df.index)
        te_col_train = pd.Series(index=X_train_df.index, dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in kf.split(X_train_df):
            tr_part = X_train_df.iloc[tr_idx]
            va_part = X_train_df.iloc[va_idx]
            y_part = y_train_enc[tr_idx]
            stats = pd.DataFrame({col: tr_part[col].astype(str), "y": y_part}).groupby(col)["y"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            te_col_train.iloc[va_idx] = va_part[col].astype(str).map(smooth).fillna(global_mean).values

        stats_full = pd.DataFrame({col: X_train_df[col].astype(str), "y": y_train_enc}).groupby(col)["y"].agg(["mean", "count"])
        smooth_full = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (stats_full["count"] + smoothing)

        X_train_enc[col] = te_col_train.values
        X_valid_enc[col] = X_valid_df[col].astype(str).map(smooth_full).fillna(global_mean).values
        X_test_enc[col] = X_test_df[col].astype(str).map(smooth_full).fillna(global_mean).values

    return X_train_enc, X_valid_enc, X_test_enc

# Prepare base frames
X_tr_2 = X_tr.copy().drop(columns=["id"], errors="ignore")
X_val_2 = X_val.copy().drop(columns=["id"], errors="ignore")
test_2 = test.copy().drop(columns=["id"], errors="ignore")

# Encode categoricals by leakage-safe target encoding
if cat_cols_2:
    X_tr_2_enc, X_val_2_enc, test_2_enc = target_encode_train_valid_test(
        X_tr_2, y_encoded[X_tr.index], X_val_2, test_2, cat_cols_2, n_splits=5, smoothing=20, seed=42
    )
else:
    X_tr_2_enc, X_val_2_enc, test_2_enc = X_tr_2.copy(), X_val_2.copy(), test_2.copy()

# Ensure all columns are numeric and aligned
X_tr_2_enc = X_tr_2_enc.reindex(columns=X_tr_2.columns, fill_value=0)
X_val_2_enc = X_val_2_enc.reindex(columns=X_tr_2.columns, fill_value=0)
test_2_enc = test_2_enc.reindex(columns=X_tr_2.columns, fill_value=0)

seeds = [42, 202, 777]
test_probas_2 = []
val_probas_2 = []
val_scores_2 = []

for seed in seeds:
    model2 = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        min_data_in_leaf=30,
        l2_leaf_reg=8,
        random_seed=seed,
        verbose=200,
        use_best_model=True
    )
    model2.fit(
        X_tr_2_enc,
        y_tr,
        eval_set=(X_val_2_enc, y_val),
        use_best_model=True
    )
    val_pred_2 = model2.predict(X_val_2_enc).ravel()
    score_2 = accuracy_score(y_val, val_pred_2)
    print(f"Solution 2 Seed={seed}, Validation Accuracy={score_2}")
    val_scores_2.append(score_2)

    raw_test_proba_2 = model2.predict_proba(test_2_enc)
    raw_val_proba_2 = model2.predict_proba(X_val_2_enc)

    model_classes_2 = list(model2.classes_)
    aligned_test_2 = np.zeros((raw_test_proba_2.shape[0], len(class_labels)), dtype=np.float64)
    aligned_val_2 = np.zeros((raw_val_proba_2.shape[0], len(class_labels)), dtype=np.float64)

    for i, cls in enumerate(model_classes_2):
        if cls in class_to_idx:
            aligned_test_2[:, class_to_idx[cls]] = raw_test_proba_2[:, i]
            aligned_val_2[:, class_to_idx[cls]] = raw_val_proba_2[:, i]

    test_probas_2.append(aligned_test_2)
    val_probas_2.append(aligned_val_2)

avg_test_proba_2 = np.mean(test_probas_2, axis=0)
avg_val_proba_2 = np.mean(val_probas_2, axis=0)
best_score_2 = float(np.mean(val_scores_2))

val_pred_2_ensemble = [class_labels[i] for i in np.argmax(avg_val_proba_2, axis=1)]
best_score_2_ensemble = accuracy_score(y_val, val_pred_2_ensemble)
print(f"Solution 2 Mean Seed Validation Accuracy: {best_score_2}")
print(f"Solution 2 Averaged Validation Performance: {best_score_2_ensemble}")

########################################
# Ensemble
########################################
# Use validation accuracies to determine simple weights
score_1 = float(best_score_1)
score_2 = float(best_score_2_ensemble)

# Safe normalized weights with a minimum contribution from each model
eps = 1e-6
w1 = max(score_1, eps)
w2 = max(score_2, eps)
w_sum = w1 + w2
w1 /= w_sum
w2 /= w_sum

print(f"Ensemble Weights -> w1: {w1}, w2: {w2}")

final_val_proba = w1 * val_proba_1 + w2 * avg_val_proba_2
final_val_pred = [class_labels[i] for i in np.argmax(final_val_proba, axis=1)]
final_validation_score = accuracy_score(y_val, final_val_pred)

print(f"Final Validation Performance: {final_validation_score}")

final_test_proba = w1 * proba_1 + w2 * avg_test_proba_2
final_test_pred_idx = np.argmax(final_test_proba, axis=1)
final_test_pred = [class_labels[i] for i in final_test_pred_idx]

submission = pd.DataFrame({
    "id": test["id"],
    "NObeyesdad": final_test_pred
})
submission.to_csv("submission.csv", index=False)
