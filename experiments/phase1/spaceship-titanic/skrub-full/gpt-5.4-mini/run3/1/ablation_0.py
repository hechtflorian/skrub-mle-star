
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

def build_graph(train_part, variant_name):
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    if variant_name == "baseline":
        X_proc = X_train.skb.apply(skrub.TableVectorizer())
    elif variant_name == "no_numeric_scale":
        X_proc = X_train.skb.apply(skrub.TableVectorizer())
    else:
        X_proc = X_train.skb.apply(skrub.TableVectorizer())

    model = HistGradientBoostingClassifier(random_state=42)
    pred = X_proc.skb.apply(model, y=y_train)
    return pred

def eval_variant(variant_name, train_part, valid_part):
    pred = build_graph(train_part, variant_name)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

results = {}
results["baseline"] = eval_variant("baseline", train_part, valid_part)
results["no_numeric_scale"] = eval_variant("no_numeric_scale", train_part, valid_part)

best_variant = max(results, key=results.get)
final_validation_score = results[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
