
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
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
    out["weekofyear"] = out["datetime"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin([7, 8, 17, 18]).astype(int)
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_dayofweek"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["cos_dayofweek"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
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

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgbm_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=random_state,
    verbose=0,
    allow_writing_files=False,
)

pred_graph_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
pred_graph_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

learner_lgbm = pred_graph_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_graph_cat.skb.make_learner(fitted=True)

valid_pred_lgbm = learner_lgbm.predict({"data": valid_part})
valid_pred_cat = learner_cat.predict({"data": valid_part})

valid_pred = 0.5 * np.asarray(valid_pred_lgbm, dtype=float) + 0.5 * np.asarray(valid_pred_cat, dtype=float)
valid_pred = np.clip(valid_pred, 0, None)

final_validation_score = mean_squared_log_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
