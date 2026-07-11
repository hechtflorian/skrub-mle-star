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
metric_label = "RMSLE"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def rmsle(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def feature_builder(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    return df


def feature_builder_no_atemp(df):
    df = feature_builder(df)
    if "atemp" in df.columns:
        df = df.drop(columns=["atemp"])
    return df


def feature_builder_drop_leakage(df):
    df = feature_builder(df)
    drop_cols = [c for c in ["casual", "registered"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def build_model():
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    predictor = build_graph(data_train)
    learner = predictor.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = rmsle(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = build_model()
    return X_train.skb.apply_func(feature_builder).skb.apply(vectorizer).skb.apply(
        model, y=y_train
    )


def no_datetime_features_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = build_model()
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def drop_atemp_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = build_model()
    return X_train.skb.apply_func(feature_builder_no_atemp).skb.apply(vectorizer).skb.apply(
        model, y=y_train
    )


def drop_leakage_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = build_model()
    return X_train.skb.apply_func(feature_builder_drop_leakage).skb.apply(vectorizer).skb.apply(
        model, y=y_train
    )


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["no_datetime_features"] = score_variant("no_datetime_features", no_datetime_features_graph)
scores["drop_atemp"] = score_variant("drop_atemp", drop_atemp_graph)
scores["drop_leakage"] = score_variant("drop_leakage", drop_leakage_graph)

baseline_score = scores["baseline"]
deltas = {k: v - baseline_score for k, v in scores.items() if k != "baseline"}
most_impact_variant = max(deltas, key=lambda k: abs(deltas[k]))
impact_direction = "improved" if deltas[most_impact_variant] < 0 else "worsened"
impact_amount = abs(deltas[most_impact_variant])

print(
    f"Most impactful part: {most_impact_variant} ({impact_direction} {metric_label} by {impact_amount})"
)