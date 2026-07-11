
import os
import glob
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

random_state = 42
target_col = "Rings"

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5

def find_csv(name):
    candidates = [
        os.path.join(".", "input", name),
        os.path.join(".", "input", "*", name),
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under ./input")

train_path = find_csv("train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgb = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgb_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=random_state,
    verbose=-1,
)

cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    verbose=0,
    random_state=random_state,
)

predictor_lgb = X_train.skb.apply(vectorizer_lgb).skb.apply(lgb_model, y=y_train)
predictor_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

learner_lgb = predictor_lgb.skb.make_learner(fitted=True)
learner_cat = predictor_cat.skb.make_learner(fitted=True)

valid_pred_lgb = np.clip(np.asarray(learner_lgb.predict({"data": valid_part})), 0, None)
valid_pred_cat = np.clip(np.asarray(learner_cat.predict({"data": valid_part})), 0, None)

valid_pred = 0.5 * valid_pred_lgb + 0.5 * valid_pred_cat
valid_pred = np.clip(np.asarray(valid_pred), 0, None)

final_validation_score = rmsle(valid_part[target_col].to_numpy(), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
