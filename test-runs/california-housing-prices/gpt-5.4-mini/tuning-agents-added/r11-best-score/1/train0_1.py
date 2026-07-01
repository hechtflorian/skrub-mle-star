
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    low_cardinality="passthrough",
    high_cardinality=skrub.StringEncoder(),
)

cat_model = CatBoostRegressor(
    iterations=3000,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    verbose=0,
    random_seed=42,
    early_stopping_rounds=100,
)

lgb_model = LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

cat_graph = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
lgb_graph = X_train.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train)

cat_learner = cat_graph.skb.make_learner(fitted=True)
lgb_learner = lgb_graph.skb.make_learner(fitted=True)

valid_pred_cat = cat_learner.predict({"data": valid_part})
valid_pred_lgb = lgb_learner.predict({"data": valid_part})

valid_pred = 0.5 * valid_pred_cat + 0.5 * valid_pred_lgb
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
