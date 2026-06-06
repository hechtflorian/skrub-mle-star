
import os
import pandas as pd
import numpy as np
import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "median_house_value"

# ---------------------------
# Solution 1 pipeline
# ---------------------------
data_1 = skrub.var("data", train_df)
X_1 = data_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_1 = data_1[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()
regressor_1 = HistGradientBoostingRegressor(random_state=0)

pred_1 = X_1.skb.apply(vectorizer_1).skb.apply(regressor_1, y=y_1)

search_1 = pred_1.skb.make_randomized_search(
    scoring="neg_root_mean_squared_error",
    n_iter=8,
    n_jobs=4,
    random_state=0,
    fitted=True,
)

if hasattr(search_1, "best_learner_") and search_1.best_learner_ is not None:
    final_learner_1 = search_1.best_learner_
elif hasattr(search_1, "best_estimator_") and search_1.best_estimator_ is not None:
    final_learner_1 = search_1.best_estimator_
else:
    final_learner_1 = pred_1.skb.make_learner(fitted=True)

# Quick holdout/preview evaluation for weighting
preview_n = min(5000, len(train_df))
preview_df = train_df.sample(n=preview_n, random_state=0) if len(train_df) > preview_n else train_df.copy()
preview_target = preview_df[target_col].to_numpy()

try:
    preview_data_1 = skrub.var("data", preview_df)
    preview_X_1 = preview_data_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    preview_y_1 = preview_data_1[target_col].skb.mark_as_y()
    quick_pred_1 = preview_X_1.skb.apply(vectorizer_1).skb.apply(regressor_1, y=preview_y_1)
    quick_learner_1 = quick_pred_1.skb.make_learner(fitted=True)
    cv_pred_1 = quick_learner_1.predict({"data": preview_df})
    rmse_1 = float(mean_squared_error(preview_target, cv_pred_1) ** 0.5)
except Exception:
    rmse_1 = float("nan")

# Predict test for solution 1
test_preds_1 = np.asarray(final_learner_1.predict({"data": test_df})).ravel()

# ---------------------------
# Solution 2 pipeline
# Kept mostly intact but independent DataOps flow
# ---------------------------
data_2 = skrub.var("data", train_df)
X_2 = data_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_2 = data_2[target_col].skb.mark_as_y()

# Slightly different inductive bias while preserving DataOps structure
vectorizer_2 = skrub.TableVectorizer()
regressor_2 = HistGradientBoostingRegressor(
    random_state=42,
    learning_rate=0.05,
    max_depth=6,
    max_iter=300,
    min_samples_leaf=20,
)

pred_2 = X_2.skb.apply(vectorizer_2).skb.apply(regressor_2, y=y_2)

search_2 = pred_2.skb.make_randomized_search(
    scoring="neg_root_mean_squared_error",
    n_iter=8,
    n_jobs=4,
    random_state=42,
    fitted=True,
)

if hasattr(search_2, "best_learner_") and search_2.best_learner_ is not None:
    final_learner_2 = search_2.best_learner_
elif hasattr(search_2, "best_estimator_") and search_2.best_estimator_ is not None:
    final_learner_2 = search_2.best_estimator_
else:
    final_learner_2 = pred_2.skb.make_learner(fitted=True)

try:
    preview_data_2 = skrub.var("data", preview_df)
    preview_X_2 = preview_data_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    preview_y_2 = preview_data_2[target_col].skb.mark_as_y()
    quick_pred_2 = preview_X_2.skb.apply(vectorizer_2).skb.apply(regressor_2, y=preview_y_2)
    quick_learner_2 = quick_pred_2.skb.make_learner(fitted=True)
    cv_pred_2 = quick_learner_2.predict({"data": preview_df})
    rmse_2 = float(mean_squared_error(preview_target, cv_pred_2) ** 0.5)
except Exception:
    rmse_2 = float("nan")

# Predict test for solution 2
test_preds_2 = np.asarray(final_learner_2.predict({"data": test_df})).ravel()

# ---------------------------
# Ensemble layer
# ---------------------------
# Inverse-RMSE weighting if possible; otherwise equal weights
valid_rmse_1 = np.isfinite(rmse_1) and rmse_1 > 0
valid_rmse_2 = np.isfinite(rmse_2) and rmse_2 > 0

if valid_rmse_1 and valid_rmse_2:
    inv_1 = 1.0 / rmse_1
    inv_2 = 1.0 / rmse_2
    w_1 = inv_1 / (inv_1 + inv_2)
    w_2 = inv_2 / (inv_1 + inv_2)
elif valid_rmse_1:
    w_1, w_2 = 0.7, 0.3
elif valid_rmse_2:
    w_1, w_2 = 0.3, 0.7
else:
    w_1, w_2 = 0.5, 0.5

# Clipped average for robustness before final blend
pred_1_clip = np.clip(test_preds_1, -1e9, 1e9)
pred_2_clip = np.clip(test_preds_2, -1e9, 1e9)

pred_matrix = np.vstack([pred_1_clip, pred_2_clip]).T
ensemble_preds = np.average(pred_matrix, axis=1, weights=[w_1, w_2])

# Validation performance on preview set using the same weighted ensemble logic
try:
    preview_preds_1 = np.asarray(final_learner_1.predict({"data": preview_df})).ravel()
    preview_preds_2 = np.asarray(final_learner_2.predict({"data": preview_df})).ravel()
    preview_ensemble = np.average(
        np.vstack([np.clip(preview_preds_1, -1e9, 1e9), np.clip(preview_preds_2, -1e9, 1e9)]).T,
        axis=1,
        weights=[w_1, w_2],
    )
    final_validation_score = float(mean_squared_error(preview_target, preview_ensemble) ** 0.5)
except Exception:
    final_validation_score = float("nan")

print(f"Final Validation Performance: {final_validation_score}")

# Write submission
submission = pd.DataFrame({"median_house_value": np.asarray(ensemble_preds).ravel()})
submission.to_csv(SUBMISSION_PATH, index=False)
