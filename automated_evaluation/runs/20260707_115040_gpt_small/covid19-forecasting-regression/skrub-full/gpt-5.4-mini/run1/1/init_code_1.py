
import os
import json
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

random_state = 42
test_size = 0.2
target_cols = ["ConfirmedCases", "Fatalities"]

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

def add_date_features(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["DayOfYear"] = out["Date"].dt.dayofyear
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype("float")
    out["DateOrdinal"] = out["Date"].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    out = out.drop(columns=["Date"])
    return out

def to_num_safe(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].replace("missing", np.nan)
            if col not in ["Province_State", "Country_Region"]:
                out[col] = pd.to_numeric(out[col], errors="ignore")
    return out

def rmsle(y_true, y_pred):
    y_true = np.maximum(np.asarray(y_true, dtype=float), 0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_cols, errors="ignore").skb.mark_as_X()
y_train = data_train[target_cols].skb.mark_as_y()

# Minimal change to fix CatBoost receiving literal "missing" in numeric features:
# clean the feature table before vectorization/model fitting while keeping the DataOps graph.
X_train = X_train.skb.apply_func(to_num_safe).skb.apply_func(add_date_features)
valid_features = add_date_features(valid_part.drop(columns=target_cols, errors="ignore"))

case_model = CatBoostRegressor(
    loss_function="MultiRMSE",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=random_state,
    verbose=0,
)

case_pred_graph = X_train.skb.apply(
    skrub.TableVectorizer(),
).skb.apply(case_model, y=y_train)

case_learner = case_pred_graph.skb.make_learner(fitted=True)
valid_pred = case_learner.predict({"data": valid_part})

# Ensure 2D predictions for MultiRMSE output
valid_pred = np.asarray(valid_pred)
if valid_pred.ndim == 1:
    valid_pred = valid_pred.reshape(-1, 1)

final_validation_score = rmsle(valid_part[target_cols].values, np.maximum(valid_pred, 0))
print(f"Final Validation Performance: {final_validation_score}")
