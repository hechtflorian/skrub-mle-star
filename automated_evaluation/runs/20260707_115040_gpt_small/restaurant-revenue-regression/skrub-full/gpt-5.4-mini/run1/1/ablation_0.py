
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

random_state = 42
test_size = 0.2
target_col = "revenue"
metric_label = "rmse"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Basic preprocessing helpers
def fe_func(df):
    out = df.copy()
    if "Open Date" in out.columns:
        out["Open Date"] = pd.to_datetime(out["Open Date"], errors="coerce")
        out["Open Year"] = out["Open Date"].dt.year
        out["Open Month"] = out["Open Date"].dt.month
        out["Open Day"] = out["Open Date"].dt.day
        out["Open DayOfWeek"] = out["Open Date"].dt.dayofweek
        out = out.drop(columns=["Open Date"])
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostRegressor(verbose=0, random_seed=random_state)
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

def no_fe_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostRegressor(verbose=0, random_seed=random_state)
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

def fe_graph(data_train):
    data_train = data_train.skb.apply_func(fe_func)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostRegressor(verbose=0, random_seed=random_state)
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)

scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["no_fe"] = score_variant("no_fe", no_fe_graph)
scores["with_fe"] = score_variant("with_fe", fe_graph)

best_variant = min(scores, key=scores.get)
final_validation_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
