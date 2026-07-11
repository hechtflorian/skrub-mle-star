
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "count"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")  # kept for script completeness

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()


def feature_builder(df):
    df = df.copy()

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    df["datetime"] = dt  # keep as true datetime so skrub routes it as datetime, not string

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek

    df["peak_period"] = pd.Series("night", index=df.index, dtype="object")
    df.loc[df["hour"].between(7, 10, inclusive="both"), "peak_period"] = "morning_peak"
    df.loc[df["hour"].between(17, 20, inclusive="both"), "peak_period"] = "evening_peak"

    if "humidity" in df.columns and "temp" in df.columns:
        df["humidity_x_temp"] = df["humidity"] * df["temp"]

    df = df.drop(columns=["datetime", "temp"], errors="ignore")
    return df


vectorizer = skrub.TableVectorizer()
model = LGBMRegressor(
    objective="poisson",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

predictor = X_train.skb.apply_func(feature_builder).skb.apply(vectorizer).skb.apply(
    model,
    y=y_train,
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = np.asarray(val_learner.predict({"data": valid_part}))
valid_pred = np.maximum(valid_pred, 0)

final_validation_score = mean_squared_log_error(
    valid_part[target_col],
    valid_pred,
) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
