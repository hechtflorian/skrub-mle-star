import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "Personality"
metric_label = "accuracy"

train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def add_social_ratios(df):
    out = df.copy()
    ratio_specs = [
        ("Time_spent_Alone", "Going_outside", "alone_per_outside"),
        ("Social_event_attendance", "Friends_circle_size", "events_per_friend"),
        ("Post_frequency", "Social_event_attendance", "posts_per_event"),
    ]
    for numer_col, denom_col, new_col in ratio_specs:
        if numer_col in out.columns and denom_col in out.columns:
            denom = out[denom_col].replace(0, np.nan)
            out[new_col] = (
                (out[numer_col] / denom)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
    return out


def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            random_state=random_state,
            verbose=0,
        ),
        y=y_train,
    )


def drop_id_graph(data_train):
    X_train = (
        data_train.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
    )
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            random_state=random_state,
            verbose=0,
        ),
        y=y_train,
    )


def no_categorical_graph(data_train):
    X_train = (
        data_train.drop(columns=target_col, errors="ignore")
        .skb.apply(skrub.DropCols(cols=["Stage_fear", "Drained_after_socializing"]))
        .skb.mark_as_X()
    )
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            random_state=random_state,
            verbose=0,
        ),
        y=y_train,
    )


def ratio_features_no_id_graph(data_train):
    data_fe = data_train.skb.apply_func(add_social_ratios)
    X_train = (
        data_fe.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
    )
    y_train = data_fe[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            random_state=random_state,
            verbose=0,
        ),
        y=y_train,
    )


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["drop_id"] = score_variant("drop_id", drop_id_graph)
scores["no_categorical"] = score_variant("no_categorical", no_categorical_graph)
scores["ratio_features_no_id"] = score_variant(
    "ratio_features_no_id", ratio_features_no_id_graph
)

baseline_score = scores["baseline"]
best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]

largest_impact_variant = max(scores, key=lambda k: abs(scores[k] - baseline_score))
largest_impact_delta = scores[largest_impact_variant] - baseline_score

print(
    f"Most impactful ablation vs baseline: {largest_impact_variant} | "
    f"delta_{metric_label}: {largest_impact_delta}"
)