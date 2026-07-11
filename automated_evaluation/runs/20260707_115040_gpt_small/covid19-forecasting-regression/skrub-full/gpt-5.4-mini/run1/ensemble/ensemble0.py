
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

random_state = 42
test_size = 0.2

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_cols = ["ConfirmedCases", "Fatalities"]


def add_date_features(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["day"] = out["Date"].dt.day
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["date_ordinal"] = out["Date"].map(pd.Timestamp.toordinal)
    out["region"] = out["Country_Region"].fillna("") + "_" + out["Province_State"].fillna("")
    out = out.drop(columns=["Date"])
    return out


def rmsle(y_true, y_pred):
    y_true = np.maximum(np.asarray(y_true), 0)
    y_pred = np.maximum(np.asarray(y_pred), 0)
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

    # Shared upstream FE layer kept intact
    X_train_fe = X_train.skb.apply_func(add_date_features)

    # Leg 1: exact original pipeline
    vectorizer_1 = skrub.TableVectorizer()
    model_1 = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
    pred_1 = X_train_fe.skb.apply(vectorizer_1).skb.apply(model_1, y=y_train)
    learner_1 = pred_1.skb.make_learner(fitted=True)
    valid_pred_1 = np.asarray(learner_1.predict({"data": valid_part}), dtype=float).ravel()

    # Leg 2: same structural pipeline, harmless merge-side diversity tweak
    # Slightly different deterministic training slice for the second leg
    train_part_2 = train_part.iloc[::2].copy() if len(train_part) > 1 else train_part.copy()
    data_train_2 = skrub.var("data", train_part_2)
    X_train_2 = data_train_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train_2 = data_train_2[target_col].skb.mark_as_y()
    X_train_2_fe = X_train_2.skb.apply_func(add_date_features)

    vectorizer_2 = skrub.TableVectorizer()
    model_2 = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=random_state + 1,
        n_jobs=1,
        verbose=-1,
    )
    pred_2 = X_train_2_fe.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train_2)
    learner_2 = pred_2.skb.make_learner(fitted=True)
    valid_pred_2 = np.asarray(learner_2.predict({"data": valid_part}), dtype=float).ravel()

    # Conservative blend
    valid_pred = 0.7 * valid_pred_1 + 0.3 * valid_pred_2

    score = rmsle(valid_part[target_col].values, valid_pred)
    scores.append(score)

final_validation_score = float(np.mean(scores))
print(f"Final Validation Performance: {final_validation_score}")
