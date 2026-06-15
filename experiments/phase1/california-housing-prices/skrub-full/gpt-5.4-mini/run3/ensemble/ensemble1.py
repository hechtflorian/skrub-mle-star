
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

target_min = train_part[target_col].min()
target_max = train_part[target_col].max()


@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    for numer_col, denom_col, out_col in [
        ("total_rooms", "households", "rooms_per_household"),
        ("total_bedrooms", "households", "bedrooms_per_household"),
        ("population", "households", "population_per_household"),
    ]:
        if numer_col in out.columns and denom_col in out.columns:
            denom = out[denom_col].replace(0, np.nan)
            out[out_col] = (
                out[numer_col] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_learner(train_part, seed):
    data_train = skrub.var("data", train_part)
    data_train = data_train.skb.apply_func(add_ratio_features)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    predictor = X_train.skb.apply(vectorizer).skb.apply(
        CatBoostRegressor(verbose=0, random_seed=seed),
        y=y_train,
    )
    return predictor.skb.make_learner(fitted=True)


def clip_pred(pred):
    return np.clip(np.asarray(pred, dtype=float), target_min, target_max)


learner_1 = build_learner(train_part, seed=42)
learner_2 = build_learner(train_part, seed=43)

valid_pred_1 = clip_pred(learner_1.predict({"data": valid_part}))
valid_pred_2 = clip_pred(learner_2.predict({"data": valid_part}))

rmse_1 = mean_squared_error(valid_part[target_col], valid_pred_1) ** 0.5
rmse_2 = mean_squared_error(valid_part[target_col], valid_pred_2) ** 0.5

w1 = 1.0 / (rmse_1 + 1e-9)
w2 = 1.0 / (rmse_2 + 1e-9)
w_sum = w1 + w2
w1 /= w_sum
w2 /= w_sum

floor = 0.25
if w1 > w2:
    w2 = max(w2, floor)
    w1 = 1.0 - w2
else:
    w1 = max(w1, floor)
    w2 = 1.0 - w1

valid_pred = w1 * valid_pred_1 + w2 * valid_pred_2
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
