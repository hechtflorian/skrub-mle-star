
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "median_house_value"

# Keep a light subsampling path for fast iteration if needed, but do not remove it.
# Use a small preview subsample only when the dataset is large enough.
if len(train_df) > 5000:
    preview_df = train_df.sample(n=5000, random_state=42)
else:
    preview_df = train_df.copy()

train_part, valid_part = train_test_split(preview_df, test_size=0.2, random_state=42)

# DataOps-first pipeline
data = skrub.var("data", train_part)

X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# Keep preprocessing simple and robust
vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

# Two candidate models
gbr = HistGradientBoostingRegressor(random_state=42)
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)

pred_gbr = X_vec.skb.apply(gbr, y=y)
pred_rf = X_vec.skb.apply(rf, y=y)

# Fit learners from DataOps graphs
learner_gbr = pred_gbr.skb.make_learner(fitted=True)
learner_rf = pred_rf.skb.make_learner(fitted=True)

X_valid = valid_part.drop(columns=target_col, errors="ignore")
y_valid = valid_part[target_col].values

pred_valid_gbr = learner_gbr.predict({"data": valid_part.drop(columns=target_col, errors="ignore").assign(**{target_col: 0})})
pred_valid_rf = learner_rf.predict({"data": valid_part.drop(columns=target_col, errors="ignore").assign(**{target_col: 0})})

rmse_gbr = mean_squared_error(y_valid, pred_valid_gbr) ** 0.5
rmse_rf = mean_squared_error(y_valid, pred_valid_rf) ** 0.5

if rmse_gbr <= rmse_rf:
    best_name = "HistGradientBoostingRegressor"
    best_model = HistGradientBoostingRegressor(random_state=42)
    final_validation_score = rmse_gbr
else:
    best_name = "RandomForestRegressor"
    best_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    final_validation_score = rmse_rf

print(f"Model selected: {best_name}")
print(f"Final Validation Performance: {final_validation_score}")

# Fit final model on full training data
full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

full_vectorizer = skrub.TableVectorizer()
X_full_vec = X_full.skb.apply(full_vectorizer)
final_pred = X_full_vec.skb.apply(best_model, y=y_full)
final_learner = final_pred.skb.make_learner(fitted=True)

# Predict on test data
test_pred = final_learner.predict({"data": test_df})

# Submission must contain only the prediction column
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
