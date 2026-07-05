
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

random_state = 42
test_size = 0.2
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

metric_label = "accuracy"

def make_base_data():
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    return X_train, y_train

def score_variant(variant_name, build_graph):
    X_train, y_train = make_base_data()
    pred = build_graph(X_train, y_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score

def baseline_graph(X_train, y_train):
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=0,
        random_seed=random_state,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

def variant_no_name_graph(X_train, y_train):
    def fe_func(df):
        out = df.copy()
        out = out.drop(columns=["Name"], errors="ignore")
        return out

    data_fe = X_train.skb.apply_func(fe_func)
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=0,
        random_seed=random_state,
    )
    return data_fe.skb.apply(vectorizer).skb.apply(model, y=y_train)

def variant_lgbm_graph(X_train, y_train):
    vectorizer = skrub.TableVectorizer()
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["drop_name"] = score_variant("drop_name", variant_no_name_graph)
scores["alt_backbone"] = score_variant("alt_backbone", variant_lgbm_graph)

best_variant = max(scores, key=scores.get)
final_validation_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
