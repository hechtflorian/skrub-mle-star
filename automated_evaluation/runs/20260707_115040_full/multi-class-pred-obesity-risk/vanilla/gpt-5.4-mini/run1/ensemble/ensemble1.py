
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

# Helper functions for rank-based ensembling
def prob_to_rank_scores(probs, scheme="inv_rank"):
    """
    Convert probability vectors to rank-based class score vectors.
    probs: shape (n_samples, n_classes)
    returns: shape (n_samples, n_classes)
    """
    probs = np.asarray(probs)
    n_samples, n_classes = probs.shape
    scores = np.zeros_like(probs, dtype=float)

    # Rank classes by descending probability per sample
    order = np.argsort(-probs, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(n_samples)[:, None]
    ranks[row_idx, order] = np.arange(1, n_classes + 1)[None, :]

    if scheme == "inv_rank":
        scores = 1.0 / ranks
    else:
        scores = 2.0 ** (-ranks)
    return scores

def get_pred_and_confidence_gap(probs):
    probs = np.asarray(probs)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top2_sorted = np.sort(top2, axis=1)
    top1 = top2_sorted[:, 1]
    top2v = top2_sorted[:, 0]
    pred = np.argmax(probs, axis=1)
    gap = top1 - top2v
    return pred, gap

best_score = -np.inf
best_model = None
best_params = None
candidate_models = []
candidate_val_scores = []
candidate_val_gaps = []

# Train candidate models and evaluate on validation
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

    val_proba = model.predict_proba(X_val_enc)
    val_pred = model.predict(X_val_enc).ravel()
    score = accuracy_score(y_val, val_pred)

    _, gap = get_pred_and_confidence_gap(val_proba)
    mean_gap = float(np.mean(gap))

    print(f"Params={params}, Validation Accuracy={score}, Mean Confidence Gap={mean_gap}")

    candidate_models.append(model)
    candidate_val_scores.append(score)
    candidate_val_gaps.append(mean_gap)

    if score > best_score:
        best_score = score
        best_model = model
        best_params = params

# Final selection of two models to ensemble:
# Keep the best and the second best candidate (if available). If only one model somehow, use it twice safely.
sorted_idx = np.argsort(candidate_val_scores)[::-1]
idx1 = int(sorted_idx[0])
idx2 = int(sorted_idx[1]) if len(sorted_idx) > 1 else int(sorted_idx[0])

model_1 = candidate_models[idx1]
model_2 = candidate_models[idx2]

acc_1 = float(candidate_val_scores[idx1])
acc_2 = float(candidate_val_scores[idx2])
gap_1 = float(candidate_val_gaps[idx1])
gap_2 = float(candidate_val_gaps[idx2])

# Normalize confidence gaps for weighting
gaps = np.array([gap_1, gap_2], dtype=float)
if np.std(gaps) > 1e-12:
    norm_gaps = (gaps - np.mean(gaps)) / (np.std(gaps) + 1e-12)
    norm_gaps = np.clip(norm_gaps, -1.0, 1.0)
else:
    norm_gaps = np.zeros_like(gaps)

raw_weights = np.array([
    acc_1 * (1.0 + norm_gaps[0]),
    acc_2 * (1.0 + norm_gaps[1])
], dtype=float)
raw_weights = np.clip(raw_weights, 1e-8, None)
weights = raw_weights / raw_weights.sum()

print(f"Selected model 1 validation accuracy: {acc_1}, gap: {gap_1}")
print(f"Selected model 2 validation accuracy: {acc_2}, gap: {gap_2}")
print(f"Ensemble weights: {weights.tolist()}")

# Validation predictions for ensemble evaluation
val_proba_1 = model_1.predict_proba(X_val_enc)
val_proba_2 = model_2.predict_proba(X_val_enc)

# Rank-based score ensemble on validation
val_rank_scores_1 = prob_to_rank_scores(val_proba_1, scheme="inv_rank")
val_rank_scores_2 = prob_to_rank_scores(val_proba_2, scheme="inv_rank")
val_merged_rank = weights[0] * val_rank_scores_1 + weights[1] * val_rank_scores_2
val_rank_pred = np.argmax(val_merged_rank, axis=1)

# Plain probability average ensemble on validation
val_prob_avg = weights[0] * val_proba_1 + weights[1] * val_proba_2
val_prob_pred = np.argmax(val_prob_avg, axis=1)

# Confidence-aware chooser: if models disagree, prefer rank ensemble in higher-confidence regime,
# otherwise use the plain average. Also break extremely close ties using the better single model.
val_gap_1 = np.max(val_proba_1, axis=1) - np.partition(val_proba_1, -2, axis=1)[:, -2]
val_gap_2 = np.max(val_proba_2, axis=1) - np.partition(val_proba_2, -2, axis=1)[:, -2]
val_conf_regime = np.maximum(val_gap_1, val_gap_2)

final_val_pred = val_rank_pred.copy()
disagree = val_rank_pred != val_prob_pred

# High confidence -> rank ensemble, Low confidence -> plain average
threshold = np.median(val_conf_regime)
final_val_pred[disagree & (val_conf_regime <= threshold)] = val_prob_pred[disagree & (val_conf_regime <= threshold)]

# Tiny tie-breaker from better single model if top two merged scores are extremely close
merged_diff = np.sort(val_merged_rank, axis=1)[:, -1] - np.sort(val_merged_rank, axis=1)[:, -2]
better_model_val_pred = np.argmax(val_proba_1 if acc_1 >= acc_2 else val_proba_2, axis=1)
final_val_pred[merged_diff < 1e-8] = better_model_val_pred[merged_diff < 1e-8]

# Convert validation labels/preds to string if needed
final_validation_score = accuracy_score(y_val, final_val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Test predictions
test_proba_1 = model_1.predict_proba(test_enc)
test_proba_2 = model_2.predict_proba(test_enc)

test_rank_scores_1 = prob_to_rank_scores(test_proba_1, scheme="inv_rank")
test_rank_scores_2 = prob_to_rank_scores(test_proba_2, scheme="inv_rank")
test_merged_rank = weights[0] * test_rank_scores_1 + weights[1] * test_rank_scores_2
test_rank_pred = np.argmax(test_merged_rank, axis=1)

test_prob_avg = weights[0] * test_proba_1 + weights[1] * test_proba_2
test_prob_pred = np.argmax(test_prob_avg, axis=1)

test_gap_1 = np.max(test_proba_1, axis=1) - np.partition(test_proba_1, -2, axis=1)[:, -2]
test_gap_2 = np.max(test_proba_2, axis=1) - np.partition(test_proba_2, -2, axis=1)[:, -2]
test_conf_regime = np.maximum(test_gap_1, test_gap_2)

final_test_pred = test_rank_pred.copy()
test_disagree = test_rank_pred != test_prob_pred
test_threshold = np.median(val_conf_regime)
final_test_pred[test_disagree & (test_conf_regime <= test_threshold)] = test_prob_pred[test_disagree & (test_conf_regime <= test_threshold)]

merged_diff_test = np.sort(test_merged_rank, axis=1)[:, -1] - np.sort(test_merged_rank, axis=1)[:, -2]
better_model_test_pred = np.argmax(test_proba_1 if acc_1 >= acc_2 else test_proba_2, axis=1)
final_test_pred[merged_diff_test < 1e-8] = better_model_test_pred[merged_diff_test < 1e-8]

# Convert numeric predictions back to original labels if necessary
if hasattr(y, "dtype") and y.dtype == object:
    classes = np.array(sorted(y.unique()))
    test_pred = classes[final_test_pred]
else:
    # Use original label ordering from CatBoost if labels are encoded as strings/objects
    try:
        test_pred = candidate_models[0].classes_[final_test_pred]
    except Exception:
        test_pred = final_test_pred

submission = pd.DataFrame({
    "id": test["id"],
    "NObeyesdad": test_pred
})
submission.to_csv("submission.csv", index=False)
