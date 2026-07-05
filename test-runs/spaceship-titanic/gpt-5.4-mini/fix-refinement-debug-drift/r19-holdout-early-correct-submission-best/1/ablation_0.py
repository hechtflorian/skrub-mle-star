
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


def score_variant(variant_name, fe_func):
    data_train = skrub.var("data", train_part)
    data_train = data_train.skb.apply_func(fe_func)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    pred = X_train.skb.apply(encoder).skb.apply(
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            verbose=0,
            random_seed=random_state,
        ),
        y=y_train,
    )
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred).astype(bool)
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score


def fe_baseline(df):
    return df.copy()


def fe_cabin_split(df):
    out = df.copy()
    cabin = out["Cabin"].astype("string")
    parts = cabin.str.split("/", expand=True)
    out["CabinDeck"] = parts[0]
    out["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
    out["CabinSide"] = parts[2]
    return out


def fe_name_flag(df):
    out = df.copy()
    out["HasName"] = out["Name"].notna().astype(int)
    return out


scores = {}
scores["baseline"] = score_variant("baseline", fe_baseline)
scores["cabin_split"] = score_variant("cabin_split", fe_cabin_split)
scores["name_flag"] = score_variant("name_flag", fe_name_flag)

best_variant = max(scores, key=scores.get)
final_validation_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
