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


def rmsle(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def add_datetime_features(df):
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["hour"] = out["datetime"].dt.hour
    out["day"] = out["datetime"].dt.day
    out["month"] = out["datetime"].dt.month
    out["year"] = out["datetime"].dt.year
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin([7, 8, 17, 18]).astype(int)
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    return out


def add_datetime_features_no_day(df):
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["hour"] = out["datetime"].dt.hour
    out["month"] = out["datetime"].dt.month
    out["year"] = out["datetime"].dt.year
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_rush_hour"] = out["hour"].isin([7, 8, 17, 18]).astype(int)
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    return out


def build_model():
    return LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def score_variant(variant_name, fe_func=None, vectorizer=None, drop_cols=None):
    data_train = skrub.var("data", train_part)
    if fe_func is not None:
        data_train = data_train.skb.apply_func(fe_func)

    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    if drop_cols is not None:
        X = X.skb.apply(skrub.DropCols(cols=drop_cols))

    if vectorizer is None:
        vectorizer = skrub.TableVectorizer()

    pred_graph = X.skb.apply(vectorizer).skb.apply(build_model(), y=y)
    learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = rmsle(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] rmsle: {score}")
    return score


scores = {}

scores["baseline"] = score_variant(
    "baseline",
    fe_func=add_datetime_features,
    vectorizer=skrub.TableVectorizer(),
)

scores["no_day_feature"] = score_variant(
    "no_day_feature",
    fe_func=add_datetime_features_no_day,
    vectorizer=skrub.TableVectorizer(),
)

scores["drop_temp_atemp"] = score_variant(
    "drop_temp_atemp",
    fe_func=add_datetime_features,
    vectorizer=skrub.TableVectorizer(),
    drop_cols=["temp", "atemp"],
)

scores["drop_datetime_fe"] = score_variant(
    "drop_datetime_fe",
    fe_func=None,
    vectorizer=skrub.TableVectorizer(),
)

best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | rmsle: {best_score}")