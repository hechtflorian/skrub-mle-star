
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")

target_col = "booking_status"
random_state = 42
test_size = 0.2
metric_fn = roc_auc_score
metric_label = "roc_auc_score"

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
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(verbose=0, random_state=random_state)
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)


def lgbm_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = LGBMClassifier(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        verbose=-1,
    )
    return X_train.skb.apply(encoder).skb.apply(model, y=y_train)


scores = {}
scores["catboost"] = score_variant("catboost", baseline_graph)
scores["lgbm"] = score_variant("lgbm", lgbm_graph)

best_variant = max(scores, key=scores.get)
final_validation_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
