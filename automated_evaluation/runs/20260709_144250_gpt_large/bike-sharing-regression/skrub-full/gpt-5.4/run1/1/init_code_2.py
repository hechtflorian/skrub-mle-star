
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
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

def add_datetime_features(df):
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["weekday"] = dt.dt.weekday
    out["log_count"] = np.log1p(out[target_col])
    return out

data_train_fe = data_train.skb.apply_func(add_datetime_features)

feature_cols = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "month", "day", "hour", "weekday"
]

X_train = data_train_fe[feature_cols].skb.mark_as_X()
y_train = data_train_fe["log_count"].skb.mark_as_y()

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=random_state,
    verbosity=0,
)

predictor = X_train.skb.apply(model, y=y_train)
val_learner = predictor.skb.make_learner(fitted=True)

valid_pred_log = np.asarray(val_learner.predict({"data": valid_part}))
valid_pred = np.expm1(valid_pred_log)
valid_pred = np.clip(valid_pred, 0, None)

final_validation_score = mean_squared_log_error(
    valid_part[target_col],
    valid_pred,
) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")
