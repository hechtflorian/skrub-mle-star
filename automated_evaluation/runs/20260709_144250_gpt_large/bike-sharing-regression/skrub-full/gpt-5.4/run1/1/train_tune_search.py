
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "count"

tune_plan = {
    "tunable_params": [
        {
            "name": "learning_rate",
            "kind": "choose_float",
            "low": 0.03,
            "high": 0.08,
            "log": True,
            "default": 0.05,
        },
        {
            "name": "num_leaves",
            "kind": "choose_int",
            "low": 24,
            "high": 40,
            "n_steps": 5,
            "default": 31,
        },
    ]
}

train_df = pd.read_csv("./input/train.csv")

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
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    return df

vectorizer = skrub.TableVectorizer()

predictor = X_train.skb.apply_func(feature_builder).skb.apply(vectorizer).skb.apply(
    LGBMRegressor(
        n_estimators=150,
        learning_rate=skrub.choose_float(
            0.03, 0.08, log=True, default=0.05, name="learning_rate"
        ),
        num_leaves=skrub.choose_int(
            24, 40, n_steps=5, default=31, name="num_leaves"
        ),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    ),
    y=y_train,
)

search = predictor.skb.make_randomized_search(
    n_iter=5, n_jobs=1, random_state=random_state, fitted=True
)
search.fit({"data": train_part})

valid_pred = np.asarray(search.best_learner_.predict({"data": valid_part}))
valid_pred = np.maximum(valid_pred, 0)

final_validation_score = mean_squared_log_error(
    valid_part[target_col],
    valid_pred,
) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

raw_values = list(search.best_params_.values())
remaining = list(raw_values)
best_params = {}

for spec in tune_plan["tunable_params"]:
    name = spec["name"]
    kind = spec["kind"]
    match = spec.get("default")

    if kind == "choose_int":
        for v in list(remaining):
            try:
                iv = int(round(float(v)))
                if spec["low"] <= iv <= spec["high"]:
                    match = iv
                    remaining.remove(v)
                    break
            except Exception:
                continue
        best_params[name] = int(match)
    elif kind == "choose_float":
        for v in list(remaining):
            try:
                fv = float(v)
                if spec["low"] <= fv <= spec["high"]:
                    match = fv
                    remaining.remove(v)
                    break
            except Exception:
                continue
        best_params[name] = float(match)
    else:
        if remaining:
            match = remaining.pop(0)
        best_params[name] = match

print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
