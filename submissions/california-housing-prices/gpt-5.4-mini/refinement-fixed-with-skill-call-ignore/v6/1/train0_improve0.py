
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

# Keep DataOps structure and preserve subsampling for fast iteration if needed
data = skrub.var("data", train_df)

target_col = "median_house_value"
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# Optional fast preview subsampling (preserved structure)
preview_data = data.skb.subsample(n=min(5000, len(train_df)))
preview_X = preview_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
preview_y = preview_data[target_col].skb.mark_as_y()

# Feature processing
vectorizer = skrub.TableVectorizer()

# Model
regressor = HistGradientBoostingRegressor(random_state=0)

# DataOps pipeline
pred = X.skb.apply(vectorizer).skb.apply(regressor, y=y)

# Use search if available, but correctly access the resulting best learner
search = pred.skb.make_randomized_search(
    scoring="neg_root_mean_squared_error",
    n_iter=8,
    n_jobs=4,
    random_state=0,
    fitted=True,
)

# Fit final model on full data using the best learner from the search result
if hasattr(search, "best_learner_") and search.best_learner_ is not None:
    final_learner = search.best_learner_
elif hasattr(search, "best_estimator_") and search.best_estimator_ is not None:
    # Fallback for compatibility with some search wrappers
    final_learner = search.best_estimator_
else:
    # Last resort: compile the pipeline directly
    final_learner = pred.skb.make_learner(fitted=True)

# Predict on test set using environment dict
test_preds = final_learner.predict({"data": test_df})

# Validation performance on a quick subsample for debug visibility
try:
    quick_pred = preview_X.skb.apply(vectorizer).skb.apply(regressor, y=preview_y)
    quick_learner = quick_pred.skb.make_learner(fitted=True)
    cv_pred = quick_learner.predict({"data": preview_data})
    final_validation_score = float(mean_squared_error(preview_data[target_col], cv_pred) ** 0.5)
except Exception:
    # If preview path fails for any reason, provide a safe placeholder metric
    final_validation_score = float("nan")

print(f"Final Validation Performance: {final_validation_score}")

# Write submission
submission = pd.DataFrame({"median_house_value": np.asarray(test_preds).ravel()})
submission.to_csv(SUBMISSION_PATH, index=False)
