
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

random_state = 42
test_size = 0.2
target_cols = ["ConfirmedCases", "Fatalities"]

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)


def add_date_features(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["day"] = out["Date"].dt.day
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype("float")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(float)
    out["date_ordinal"] = out["Date"].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    out["region"] = out["Country_Region"].fillna("").astype(str) + "_" + out["Province_State"].fillna("").astype(str)
    out = out.drop(columns=["Date"])
    return out


def to_num_safe(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].replace("missing", np.nan)
            if col not in ["Province_State", "Country_Region", "region"]:
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

scores = []

for target_col in target_cols:
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    X_train_fe = X_train.skb.apply_func(to_num_safe).skb.apply_func(add_date_features)

    vectorizer_lgb = skrub.TableVectorizer()
    model_lgb = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
    pred_lgb = X_train_fe.skb.apply(vectorizer_lgb).skb.apply(model_lgb, y=y_train)
    learner_lgb = pred_lgb.skb.make_learner(fitted=True)

    vectorizer_cat = skrub.TableVectorizer()
    model_cat = CatBoostRegressor(
        loss_function="RMSE",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        verbose=0,
    )
    pred_cat = X_train_fe.skb.apply(vectorizer_cat).skb.apply(model_cat, y=y_train)
    learner_cat = pred_cat.skb.make_learner(fitted=True)

    valid_pred_lgb = np.asarray(learner_lgb.predict({"data": valid_part}), dtype=float).ravel()
    valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part}), dtype=float).ravel()

    valid_pred = 0.6 * np.maximum(valid_pred_lgb, 0) + 0.4 * np.maximum(valid_pred_cat, 0)
    score = rmsle(valid_part[target_col].values, valid_pred)
    scores.append(score)

final_validation_score = float(np.mean(scores))
print(f"Final Validation Performance: {final_validation_score}")
