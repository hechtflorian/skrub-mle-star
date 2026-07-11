
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

    def add_date_preprocessing(df):
        out = df.copy()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out["Date_year"] = out["Date"].dt.year
            out["Date_month"] = out["Date"].dt.month
            out["Date_day"] = out["Date"].dt.day
            out["Date_dayofweek"] = out["Date"].dt.dayofweek
            out["Date_quarter"] = out["Date"].dt.quarter
            out["Date_is_month_start"] = out["Date"].dt.is_month_start.astype(float)
            out["Date_is_month_end"] = out["Date"].dt.is_month_end.astype(float)
        return out

    vectorizer = skrub.TableVectorizer(
        high_cardinality=skrub.StringEncoder(),
    )
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )

    pred = (
        X_train.skb.apply_func(add_date_preprocessing)
        .skb.apply(vectorizer)
        .skb.apply(model, y=y_train)
    )

    val_learner = pred.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})
    score = rmsle(valid_part[target_col].values, valid_pred)
    scores.append(score)


final_validation_score = float(np.mean(scores))
print(f"Final Validation Performance: {final_validation_score}")
