
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph on train partition only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Shared preprocessing
vectorizer = skrub.TableVectorizer()

# Base model: CatBoost
cat_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    random_seed=42,
    verbose=0,
)

# Reference model: LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Build two DataOps pipelines
cat_pred_graph = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
lgbm_pred_graph = X_train.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_train)

# Fit on holdout training partition only
cat_learner = cat_pred_graph.skb.make_learner(fitted=True)
lgbm_learner = lgbm_pred_graph.skb.make_learner(fitted=True)

cat_valid_pred = cat_learner.predict({"data": valid_part})
lgbm_valid_pred = lgbm_learner.predict({"data": valid_part})

# Simple ensemble
valid_pred = 0.5 * cat_valid_pred + 0.5 * lgbm_valid_pred

# Evaluation metric
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: fit on full train and predict test
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

cat_full_graph = X_full.skb.apply(vectorizer).skb.apply(cat_model, y=y_full)
lgbm_full_graph = X_full.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_full)

cat_full_learner = cat_full_graph.skb.make_learner(fitted=True)
lgbm_full_learner = lgbm_full_graph.skb.make_learner(fitted=True)

cat_test_pred = cat_full_learner.predict({"data": test_df})
lgbm_test_pred = lgbm_full_learner.predict({"data": test_df})

test_pred = 0.5 * cat_test_pred + 0.5 * lgbm_test_pred

submission = pd.DataFrame({target_col: test_pred})
submission.to_csv("submission.csv", index=False)
