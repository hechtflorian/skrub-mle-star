
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Shared preprocessing
vectorizer = skrub.TableVectorizer()

# Base model: LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    random_state=42,
    n_jobs=-1,
)

lgbm_pred_graph = X_train.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_train)
lgbm_learner = lgbm_pred_graph.skb.make_learner(fitted=True)
lgbm_valid_pred = lgbm_learner.predict({"data": valid_part})

# Reference model: RandomForest
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

rf_pred_graph = X_train.skb.apply(vectorizer).skb.apply(rf_model, y=y_train)
rf_learner = rf_pred_graph.skb.make_learner(fitted=True)
rf_valid_pred = rf_learner.predict({"data": valid_part})

# Simple ensemble of the two models
valid_pred = 0.7 * np.asarray(lgbm_valid_pred) + 0.3 * np.asarray(rf_valid_pred)
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Submission-stage refit on full training data and predict test
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_lgbm_graph = X_full.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_full)
full_rf_graph = X_full.skb.apply(vectorizer).skb.apply(rf_model, y=y_full)

full_lgbm_learner = full_lgbm_graph.skb.make_learner(fitted=True)
full_rf_learner = full_rf_graph.skb.make_learner(fitted=True)

test_pred_lgbm = full_lgbm_learner.predict({"data": test_df})
test_pred_rf = full_rf_learner.predict({"data": test_df})
test_pred = 0.7 * np.asarray(test_pred_lgbm) + 0.3 * np.asarray(test_pred_rf)

submission = pd.DataFrame({target_col: test_pred})
submission.to_csv("submission.csv", index=False)
