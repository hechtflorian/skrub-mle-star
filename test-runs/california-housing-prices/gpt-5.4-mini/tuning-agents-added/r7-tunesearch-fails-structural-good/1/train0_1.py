
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = "./input/train.csv"
test_path = "./input/test.csv"

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

vectorizer_cb = skrub.TableVectorizer()
vectorizer_lgb = skrub.TableVectorizer()

cat_model = CatBoostRegressor(
    iterations=5000,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200,
    allow_writing_files=False,
)

lgb_model = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pred_cb = X_train.skb.apply(vectorizer_cb).skb.apply(cat_model, y=y_train)
pred_lgb = X_train.skb.apply(vectorizer_lgb).skb.apply(lgb_model, y=y_train)

val_learner_cb = pred_cb.skb.make_learner(fitted=True)
val_learner_lgb = pred_lgb.skb.make_learner(fitted=True)

valid_pred_cb = val_learner_cb.predict({"data": valid_part})
valid_pred_lgb = val_learner_lgb.predict({"data": valid_part})

final_pred = 0.5 * np.asarray(valid_pred_cb) + 0.5 * np.asarray(valid_pred_lgb)
final_validation_score = mean_squared_error(valid_part[target_col], final_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
