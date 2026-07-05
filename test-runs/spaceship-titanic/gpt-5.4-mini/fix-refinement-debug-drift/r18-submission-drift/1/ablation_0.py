
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

random_state = 42
test_size = 0.2
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def build_graph(data_train, variant_name):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    if variant_name == "baseline":
        encoder = skrub.TableVectorizer()
        model = LGBMClassifier(
            random_state=random_state,
            n_estimators=200,
            learning_rate=0.05,
            verbose=-1,
        )
        return X.skb.apply(encoder).skb.apply(model, y=y)

    if variant_name == "alt_vectorizer":
        encoder = skrub.TableVectorizer(high_cardinality="drop")
        model = LGBMClassifier(
            random_state=random_state,
            n_estimators=200,
            learning_rate=0.05,
            verbose=-1,
        )
        return X.skb.apply(encoder).skb.apply(model, y=y)

    if variant_name == "more_trees":
        encoder = skrub.TableVectorizer()
        model = LGBMClassifier(
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.03,
            verbose=-1,
        )
        return X.skb.apply(encoder).skb.apply(model, y=y)

    encoder = skrub.TableVectorizer()
    model = LGBMClassifier(
        random_state=random_state,
        n_estimators=200,
        learning_rate=0.05,
        verbose=-1,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

def score_variant(variant_name):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train, variant_name)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred)
    if valid_pred.dtype != bool:
        valid_pred = valid_pred.astype(str)
        valid_pred = np.array([v == "True" for v in valid_pred], dtype=bool)
    final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy_score: {final_validation_score}")
    return final_validation_score

scores = {}
scores["baseline"] = score_variant("baseline")
scores["alt_vectorizer"] = score_variant("alt_vectorizer")
scores["more_trees"] = score_variant("more_trees")

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy_score: {best_score}")
print(f"Final Validation Performance: {best_score}")
