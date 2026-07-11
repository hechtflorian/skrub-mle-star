
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()


def feature_builder_leg1(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    return df


def feature_builder_leg2(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return df


def feature_builder_leg3(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["quarter"] = dt.dt.quarter
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["is_workhour"] = df["hour"].between(7, 19).astype(int)
    return df


def rmsle(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def log_blend(preds, weights):
    log_preds = [np.log1p(np.maximum(np.asarray(p, dtype=float), 0)) for p in preds]
    out = np.zeros_like(log_preds[0], dtype=float)
    for w, lp in zip(weights, log_preds):
        out += w * lp
    return np.maximum(np.expm1(out), 0)


vectorizer_1 = skrub.TableVectorizer()
model_1 = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.04707029531351466,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)
predictor_1 = (
    X_train.skb.apply_func(feature_builder_leg1)
    .skb.apply(vectorizer_1)
    .skb.apply(model_1, y=y_train)
)
learner_1 = predictor_1.skb.make_learner(fitted=True)

vectorizer_2 = skrub.TableVectorizer()
model_2 = RandomForestRegressor(
    n_estimators=400,
    max_depth=18,
    min_samples_leaf=2,
    random_state=random_state,
    n_jobs=-1,
)
predictor_2 = (
    X_train.skb.apply_func(feature_builder_leg2)
    .skb.apply(vectorizer_2)
    .skb.apply(model_2, y=y_train)
)
learner_2 = predictor_2.skb.make_learner(fitted=True)

vectorizer_3 = skrub.TableVectorizer()
model_3 = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=22,
    min_samples_leaf=1,
    random_state=random_state,
    n_jobs=-1,
)
predictor_3 = (
    X_train.skb.apply_func(feature_builder_leg3)
    .skb.apply(vectorizer_3)
    .skb.apply(model_3, y=y_train)
)
learner_3 = predictor_3.skb.make_learner(fitted=True)

p1 = np.maximum(np.asarray(learner_1.predict({"data": valid_part}), dtype=float).ravel(), 0)
p2 = np.maximum(np.asarray(learner_2.predict({"data": valid_part}), dtype=float).ravel(), 0)
p3 = np.maximum(np.asarray(learner_3.predict({"data": valid_part}), dtype=float).ravel(), 0)

cand_A = log_blend([p1, p2, p3], (0.65, 0.25, 0.10))
cand_B = log_blend([p1, p2, p3], (0.25, 0.55, 0.20))
cand_C = log_blend([p1, p2, p3], (0.25, 0.20, 0.55))
cand_D = np.maximum(
    np.expm1(
        np.median(
            np.vstack([np.log1p(p1), np.log1p(p2), np.log1p(p3)]),
            axis=0,
        )
    ),
    0,
)

candidates = [cand_A, cand_B, cand_C, cand_D]
y_valid = valid_part[target_col].to_numpy()

global_scores = [rmsle(y_valid, cand) for cand in candidates]
global_best_idx = int(np.argmin(global_scores))

dt_valid = pd.to_datetime(valid_part["datetime"])
is_weekend = (dt_valid.dt.dayofweek >= 5).to_numpy()
is_weekday = ~is_weekend

final_pred = np.zeros(len(valid_part), dtype=float)

groups = {
    "weekday": is_weekday,
    "weekend": is_weekend,
}

min_bucket_size = 30

for _, mask in groups.items():
    idx = np.where(mask)[0]
    if len(idx) == 0:
        continue
    if len(idx) < min_bucket_size:
        best_idx = global_best_idx
    else:
        group_scores = [rmsle(y_valid[idx], cand[idx]) for cand in candidates]
        best_idx = int(np.argmin(group_scores))
    final_pred[idx] = candidates[best_idx][idx]

final_pred = np.maximum(final_pred, 0)

final_validation_score = mean_squared_log_error(
    valid_part[target_col],
    final_pred,
) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
