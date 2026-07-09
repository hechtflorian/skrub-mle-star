
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

random_state = 42
test_size = 0.2
target_col = "count"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def add_datetime_features(df):
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    out["hour"] = dt.dt.hour
    out["day"] = dt.dt.day
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out = out.drop(columns=["datetime"], errors="ignore")
    return out

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_datetime_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

pred = X_train.skb.apply(
    skrub.TableVectorizer(),
).skb.apply(
    RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    ),
    y=y_train,
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = np.asarray(val_learner.predict({"data": valid_part}), dtype=float)
valid_pred = np.clip(valid_pred, 0, None)

final_validation_score = mean_squared_log_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
