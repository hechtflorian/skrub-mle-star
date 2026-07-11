import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from skrub import DropCols
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "booking_status"
metric_label = "ROC-AUC"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def build_baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def build_drop_id_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    X_train = X_train.skb.apply(DropCols(cols=["id"]))
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def add_ratio_features(df):
    out = df.copy()

    if {"no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"}.issubset(out.columns):
        denom = (
            out["no_of_previous_bookings_not_canceled"] + out["no_of_previous_cancellations"]
        ).replace(0, np.nan)
        out["previous_cancellation_ratio"] = (
            out["no_of_previous_cancellations"] / denom
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if {"no_of_weekend_nights", "no_of_week_nights"}.issubset(out.columns):
        total_nights = (
            out["no_of_weekend_nights"] + out["no_of_week_nights"]
        ).replace(0, np.nan)
        out["weekend_night_ratio"] = (
            out["no_of_weekend_nights"] / total_nights
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["total_nights"] = (
            out["no_of_weekend_nights"] + out["no_of_week_nights"]
        )

    return out


def build_with_ratio_features_graph(data_train):
    data_train = data_train.skb.apply_func(add_ratio_features)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def build_drop_redundant_prev_non_canceled_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    X_train = X_train.skb.apply(DropCols(cols=["no_of_previous_bookings_not_canceled"]))
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    predictor = build_graph(data_train)
    val_learner = predictor.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict_proba({"data": valid_part})[:, 1]
    score = roc_auc_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


scores = {}
scores["baseline"] = score_variant("baseline", build_baseline_graph)
scores["drop_id"] = score_variant("drop_id", build_drop_id_graph)
scores["add_ratio_features"] = score_variant("add_ratio_features", build_with_ratio_features_graph)
scores["drop_prev_non_canceled"] = score_variant(
    "drop_prev_non_canceled", build_drop_redundant_prev_non_canceled_graph
)

baseline_score = scores["baseline"]
best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]

if best_variant == "baseline":
    biggest_message = "baseline is strongest; tested ablations reduce performance or do not help"
else:
    contribution = best_score - baseline_score
    biggest_message = f"{best_variant} contributes the most with delta={contribution:.6f} vs baseline"

print(biggest_message)