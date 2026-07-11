
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

    leakage_cols = [col for col in ["casual", "registered"] if col in df.columns]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)

    dt = pd.to_datetime(df["datetime"])

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    df["is_commute_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_workday_commute"] = (
        (~df["dayofweek"].isin([5, 6])) & (df["hour"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)
    df["holiday_commute"] = (
        df.get("holiday", 0).astype(int) * df["is_commute_hour"]
        if "holiday" in df.columns
        else 0
    )
    df["workingday_commute"] = (
        df.get("workingday", 0).astype(int) * df["is_commute_hour"]
        if "workingday" in df.columns
        else 0
    )

    return df

vectorizer = skrub.TableVectorizer()
model = LGBMRegressor(
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
