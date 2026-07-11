
import os
import glob
import numpy as np
import pandas as pd
import skrub
from skrub import DropCols
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

random_state = 42
target_col = "Rings"
metric_label = "RMSLE"


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def find_csv(name):
    candidates = [
        os.path.join(".", "input", name),
        os.path.join(".", "input", "*", name),
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under ./input")


train_path = find_csv("train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def make_model():
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=random_state,
        verbose=-1,
    )


def add_ratio_features(df):
    out = df.copy()
    if {"Length", "Diameter"}.issubset(out.columns):
        denom = out["Diameter"].replace(0, np.nan)
        out["Length_per_Diameter"] = (
            (out["Length"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    if {"Whole weight", "Shell weight"}.issubset(out.columns):
        denom = out["Shell weight"].replace(0, np.nan)
        out["Whole_to_Shell_weight"] = (
            (out["Whole weight"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    if {"Whole weight.1", "Whole weight.2"}.issubset(out.columns):
        denom = out["Whole weight.2"].replace(0, np.nan)
        out["Weight1_per_Weight2"] = (
            (out["Whole weight.1"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    return out


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.clip(np.asarray(valid_pred), 0, None)
    score = rmsle(valid_part[target_col].to_numpy(), valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def baseline_graph(data_train):
    X = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(make_model(), y=y)


def drop_id_graph(data_train):
    X = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    X = X.skb.apply(DropCols(cols=["id"]))
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(make_model(), y=y)


def drop_redundant_weight_graph(data_train):
    X = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    X = X.skb.apply(DropCols(cols=["Whole weight.2"]))
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(make_model(), y=y)


def add_ratios_graph(data_train):
    data_fe = data_train.skb.apply_func(add_ratio_features)
    X = data_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data_fe[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X.skb.apply(vectorizer).skb.apply(make_model(), y=y)


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["drop_id"] = score_variant("drop_id", drop_id_graph)
scores["drop_redundant_weight2"] = score_variant("drop_redundant_weight2", drop_redundant_weight_graph)
scores["add_ratios"] = score_variant("add_ratios", add_ratios_graph)

baseline_score = scores["baseline"]
best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
largest_effect_variant = max(scores, key=lambda k: abs(scores[k] - baseline_score))
largest_effect_value = scores[largest_effect_variant] - baseline_score

print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
print(f"Part contributing most vs baseline: {largest_effect_variant} | delta_{metric_label}: {largest_effect_value}")
