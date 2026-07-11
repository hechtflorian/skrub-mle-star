
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

random_state = 42
test_size = 0.2
target_col = "yield"
metric_fn = mean_absolute_error
metric_label = "MAE"

train_df = pd.read_csv("./input/train.csv")

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
    score = metric_fn(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def baseline_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        verbose=0,
        random_seed=random_state,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)


def xgb_variant_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        random_state=random_state,
        verbosity=0,
        n_jobs=1,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)


def catboost_alt_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=8,
        loss_function="MAE",
        verbose=0,
        random_seed=random_state,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)


scores = {}
scores["baseline_catboost"] = score_variant("baseline_catboost", baseline_graph)
scores["xgb_variant"] = score_variant("xgb_variant", xgb_variant_graph)
scores["catboost_alt"] = score_variant("catboost_alt", catboost_alt_graph)

best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
final_validation_score = best_score
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
print(f"Final Validation Performance: {final_validation_score}")
