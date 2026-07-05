
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def build_graph(data_train, model):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

def score_variant(variant_name, model):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train, model)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred)
    if valid_pred.dtype != bool:
        valid_pred = valid_pred.astype(int) if np.issubdtype(valid_pred.dtype, np.number) else valid_pred
    if valid_pred.dtype != bool and valid_pred.ndim == 2:
        valid_pred = np.argmax(valid_pred, axis=1)
    if valid_pred.dtype == bool:
        valid_labels = valid_pred
    else:
        valid_labels = valid_pred
    score = accuracy_score(valid_part[target_col], valid_labels)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

scores = {}
scores["catboost"] = score_variant(
    "catboost",
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        verbose=0,
    ),
)
scores["lightgbm"] = score_variant(
    "lightgbm",
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    ),
)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {best_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

best_model = (
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        verbose=0,
    )
    if best_variant == "catboost"
    else LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
)

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(best_model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

valid_pred = full_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)
if valid_pred.ndim == 2:
    valid_pred = np.argmax(valid_pred, axis=1)
if valid_pred.dtype != bool:
    if np.issubdtype(valid_pred.dtype, np.number):
        valid_pred = valid_pred.astype(int)
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred)
if test_pred.ndim == 2:
    test_pred = np.argmax(test_pred, axis=1)
if test_pred.dtype != bool:
    if np.issubdtype(test_pred.dtype, np.number):
        test_pred = test_pred.astype(int)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": np.where(test_pred.astype(bool), True, False),
    }
)
submission.to_csv("submission.csv", index=False)
