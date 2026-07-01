
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TARGET = "median_house_value"
RANDOM_STATE = 42

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

vectorizer = skrub.TableVectorizer(
    low_cardinality="passthrough",
    high_cardinality=skrub.StringEncoder(n_components=20),
)

# Model 1: CatBoost
data_train_cb = skrub.var("data", train_part)
X_train_cb = data_train_cb.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_train_cb = data_train_cb[TARGET].skb.mark_as_y()

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.03,
    iterations=5000,
    random_seed=RANDOM_STATE,
    verbose=200,
)

cat_pred_graph = X_train_cb.skb.apply(vectorizer).skb.apply(cat_model, y=y_train_cb)
cat_val_learner = cat_pred_graph.skb.make_learner(fitted=True)
cat_valid_pred = cat_val_learner.predict({"data": valid_part})

# Model 2: LightGBM
data_train_lgb = skrub.var("data", train_part)
X_train_lgb = data_train_lgb.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_train_lgb = data_train_lgb[TARGET].skb.mark_as_y()

lgb_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
)

lgb_pred_graph = X_train_lgb.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train_lgb)
lgb_val_learner = lgb_pred_graph.skb.make_learner(fitted=True)
lgb_valid_pred = lgb_val_learner.predict({"data": valid_part})

# Simple ensemble
valid_pred = 0.5 * cat_valid_pred + 0.5 * lgb_valid_pred
final_validation_score = mean_squared_error(valid_part[TARGET], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Block 2: full train for test predictions
data_full_cb = skrub.var("data", train_df)
X_full_cb = data_full_cb.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_full_cb = data_full_cb[TARGET].skb.mark_as_y()

cat_full_graph = X_full_cb.skb.apply(vectorizer).skb.apply(cat_model, y=y_full_cb)
cat_full_learner = cat_full_graph.skb.make_learner(fitted=True)
cat_test_pred = cat_full_learner.predict({"data": test_df})

data_full_lgb = skrub.var("data", train_df)
X_full_lgb = data_full_lgb.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_full_lgb = data_full_lgb[TARGET].skb.mark_as_y()

lgb_full_graph = X_full_lgb.skb.apply(vectorizer).skb.apply(lgb_model, y=y_full_lgb)
lgb_full_learner = lgb_full_graph.skb.make_learner(fitted=True)
lgb_test_pred = lgb_full_learner.predict({"data": test_df})

test_pred = 0.5 * cat_test_pred + 0.5 * lgb_test_pred

submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
