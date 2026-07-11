
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

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)

# Leg 1 keeps the original feature view exactly as-is
X_train_leg1 = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_leg1 = data_train[target_col].skb.mark_as_y()

def feature_builder(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    return df

# Leg 3 uses a richer calendar feature view, still same DataOps flow
def feature_builder_richer(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    try:
        df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    except Exception:
        df["weekofyear"] = dt.dt.weekofyear.astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    return df

# Leg 2 uses the same original feature view but trains on log1p(target)
y_train_leg2 = data_train[target_col].skb.apply_func(np.log1p).skb.mark_as_y()

base_model_params = dict(
    n_estimators=300,
    learning_rate=0.04707029531351466,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

# Leg 1: original solution unchanged in structure
vectorizer_1 = skrub.TableVectorizer()
model_1 = LGBMRegressor(**base_model_params)
predictor_1 = X_train_leg1.skb.apply_func(feature_builder).skb.apply(vectorizer_1).skb.apply(
    model_1,
    y=y_train_leg1,
)
learner_1 = predictor_1.skb.make_learner(fitted=True)

# Leg 2: same structure, log-target twin
vectorizer_2 = skrub.TableVectorizer()
model_2 = LGBMRegressor(**base_model_params)
predictor_2 = X_train_leg1.skb.apply_func(feature_builder).skb.apply(vectorizer_2).skb.apply(
    model_2,
    y=y_train_leg2,
)
learner_2 = predictor_2.skb.make_learner(fitted=True)

# Leg 3: same structure, richer datetime-derived features
vectorizer_3 = skrub.TableVectorizer()
model_3 = LGBMRegressor(**base_model_params)
predictor_3 = X_train_leg1.skb.apply_func(feature_builder_richer).skb.apply(vectorizer_3).skb.apply(
    model_3,
    y=y_train_leg1,
)
learner_3 = predictor_3.skb.make_learner(fitted=True)

# Collect per-leg validation predictions
valid_pred_1 = np.asarray(learner_1.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_1 = np.maximum(valid_pred_1, 0)

valid_pred_2_log = np.asarray(learner_2.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_2 = np.expm1(valid_pred_2_log)
valid_pred_2 = np.maximum(valid_pred_2, 0)

valid_pred_3 = np.asarray(learner_3.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_3 = np.maximum(valid_pred_3, 0)

# Lightweight holdout weight search in log space
weight_grid = [
    [1.0, 0.0, 0.0],
    [0.7, 0.3, 0.0],
    [0.6, 0.2, 0.2],
    [0.5, 0.25, 0.25],
    [0.4, 0.4, 0.2],
]

best_score = float("inf")
best_pred = None

for w1, w2, w3 in weight_grid:
    blended = np.expm1(
        w1 * np.log1p(valid_pred_1) +
        w2 * np.log1p(valid_pred_2) +
        w3 * np.log1p(valid_pred_3)
    )
    blended = np.maximum(blended, 0)
    score = mean_squared_log_error(valid_part[target_col], blended) ** 0.5
    if score < best_score:
        best_score = score
        best_pred = blended

final_validation_score = mean_squared_log_error(
    valid_part[target_col],
    best_pred,
) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")
