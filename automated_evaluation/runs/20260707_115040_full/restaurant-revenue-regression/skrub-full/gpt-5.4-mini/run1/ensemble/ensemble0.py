
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

random_state = 0
target_col = "revenue"

train_df = pd.read_csv("./input/train.csv")

if "Open Date" in train_df.columns:
    train_df["Open Date"] = pd.to_datetime(train_df["Open Date"], errors="coerce")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def add_tiny_datetime_features(df):
    out = df.copy()
    if "Open Date" in out.columns:
        dt = pd.to_datetime(out["Open Date"], errors="coerce")
        out["Open Date_year"] = dt.dt.year
        out["Open Date_month"] = dt.dt.month
        out["Open Date_dayofweek"] = dt.dt.dayofweek
        out["Open Date_ordinal"] = dt.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    return out

def fit_leg(train_frame, model_seed, use_datetime_features=False):
    data_train = skrub.var("data", train_frame)
    if use_datetime_features:
        data_train = data_train.skb.apply_func(add_tiny_datetime_features)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=model_seed,
        verbose=0,
    )
    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred.skb.make_learner(fitted=True)
    return learner

def predict_learner(learner, df, use_datetime_features=False):
    env = {"data": df}
    return np.asarray(learner.predict(env), dtype=float).ravel()

# Leg 1: exact original pipeline
learner_1 = fit_leg(train_part, model_seed=random_state, use_datetime_features=False)
valid_pred_1 = predict_learner(learner_1, valid_part, use_datetime_features=False)

# Leg 2: minimal structural variant with tiny deterministic datetime expansion
learner_2 = fit_leg(train_part, model_seed=random_state + 1, use_datetime_features=True)
valid_pred_2 = predict_learner(learner_2, valid_part, use_datetime_features=True)

# Blend
valid_pred = 0.6 * valid_pred_1 + 0.4 * valid_pred_2

final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
