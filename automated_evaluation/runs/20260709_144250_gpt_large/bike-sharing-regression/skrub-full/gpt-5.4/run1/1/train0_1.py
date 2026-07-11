
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "count"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)

def feature_builder(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["weekday"] = dt.dt.weekday
    df["log_count"] = np.log1p(df[target_col])
    return df

data_train_fe = data_train.skb.apply_func(feature_builder)

# Base solution leg: TableVectorizer + LightGBM on raw target
X_train_lgb = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_lgb = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

predictor_lgb = X_train_lgb.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train_lgb)
learner_lgb = predictor_lgb.skb.make_learner(fitted=True)

# Reference solution leg: engineered numeric features + XGBoost on log target
feature_cols = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "month", "day", "hour", "weekday"
]

X_train_xgb = data_train_fe[feature_cols].skb.mark_as_X()
y_train_xgb = data_train_fe["log_count"].skb.mark_as_y()

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=random_state,
    verbosity=0,
)

predictor_xgb = X_train_xgb.skb.apply(xgb_model, y=y_train_xgb)
learner_xgb = predictor_xgb.skb.make_learner(fitted=True)

valid_pred_lgb = np.asarray(learner_lgb.predict({"data": valid_part}), dtype=float)
valid_pred_lgb = np.clip(valid_pred_lgb, 0, None)

valid_pred_log_xgb = np.asarray(learner_xgb.predict({"data": valid_part}), dtype=float)
valid_pred_xgb = np.expm1(valid_pred_log_xgb)
valid_pred_xgb = np.clip(valid_pred_xgb, 0, None)

valid_pred = 0.5 * valid_pred_lgb + 0.5 * valid_pred_xgb
valid_pred = np.clip(valid_pred, 0, None)

final_validation_score = mean_squared_log_error(
    valid_part[target_col],
    valid_pred,
) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")
