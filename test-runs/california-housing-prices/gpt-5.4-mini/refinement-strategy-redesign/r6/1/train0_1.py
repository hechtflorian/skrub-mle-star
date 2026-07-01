
import pandas as pd
import numpy as np
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TARGET = "median_house_value"
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

# ----------------------------
# DataOps-first preprocessing
# ----------------------------
data = skrub.var("data", train_part)
X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

# ----------------------------
# Model 1: CatBoost
# ----------------------------
cat_model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    random_seed=42,
    verbose=200,
)

cat_pred_graph = X_vec.skb.apply(cat_model, y=y)
cat_learner = cat_pred_graph.skb.make_learner(fitted=True)
cat_valid_pred = cat_learner.predict({"data": valid_part})

# ----------------------------
# Model 2: LightGBM
# ----------------------------
lgb_model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

lgb_pred_graph = X_vec.skb.apply(lgb_model, y=y)
lgb_learner = lgb_pred_graph.skb.make_learner(fitted=True)
lgb_valid_pred = lgb_learner.predict({"data": valid_part})

# ----------------------------
# Simple ensemble on hold-out
# Use equal-weight average for robustness
# ----------------------------
valid_pred = 0.5 * cat_valid_pred + 0.5 * lgb_valid_pred
rmse = mean_squared_error(valid_part[TARGET], valid_pred) ** 0.5
print(f"Final Validation Performance: {rmse}")

# ----------------------------
# Train on full data for test prediction
# ----------------------------
full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
full_y = full_data[TARGET].skb.mark_as_y()
full_X_vec = full_X.skb.apply(vectorizer)

# Full CatBoost
full_cat_graph = full_X_vec.skb.apply(
    CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=3000,
        random_seed=42,
        verbose=200,
    ),
    y=full_y,
)
full_cat_learner = full_cat_graph.skb.make_learner(fitted=True)
cat_test_pred = full_cat_learner.predict({"data": test_df})

# Full LightGBM
full_lgb_graph = full_X_vec.skb.apply(
    lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
    y=full_y,
)
full_lgb_learner = full_lgb_graph.skb.make_learner(fitted=True)
lgb_test_pred = full_lgb_learner.predict({"data": test_df})

# Ensemble test prediction
test_pred = 0.5 * cat_test_pred + 0.5 * lgb_test_pred

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
