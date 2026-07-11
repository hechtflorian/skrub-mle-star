
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")

random_state = 42
target_col = "count"

def add_datetime_features(df):
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["hour"] = out["datetime"].dt.hour
    out["day"] = out["datetime"].dt.day
    out["month"] = out["datetime"].dt.month
    out["year"] = out["datetime"].dt.year
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin([7, 8, 17, 18]).astype(int)
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_datetime_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)

valid_pred = learner.predict({"data": valid_part})
valid_pred = np.clip(valid_pred, 0, None)

final_validation_score = mean_squared_log_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
