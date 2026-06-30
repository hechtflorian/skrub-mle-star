
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"

# Keep the same holdout structure and avoid leakage
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def build_graph(data_train, variant_name):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    if variant_name == "baseline":
        X_feat = X
    elif variant_name == "no_name":
        def fe_func(df):
            out = df.copy()
            out = out.drop(columns=["Name"], errors="ignore")
            return out
        X_feat = X.skb.apply_func(fe_func)
    elif variant_name == "drop_cabin":
        def fe_func(df):
            out = df.copy()
            out = out.drop(columns=["Cabin"], errors="ignore")
            return out
        X_feat = X.skb.apply_func(fe_func)
    else:
        X_feat = X

    vectorizer = skrub.TableVectorizer()
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    return X_feat.skb.apply(vectorizer).skb.apply(model, y=y)

def score_variant(variant_name):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train, variant_name)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

scores = {}
scores["baseline"] = score_variant("baseline")
scores["no_name"] = score_variant("no_name")
scores["drop_cabin"] = score_variant("drop_cabin")

best_variant = max(scores, key=scores.get)
final_validation_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {final_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")
