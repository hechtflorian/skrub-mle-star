
import os
import json
import warnings
import numpy as np
import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(INPUT_DIR, "submission.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "Transported"

def build_voting_classifier():
    cat_model_local = CatBoostClassifier(
        iterations=180,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        random_seed=42,
        verbose=0,
    )
    lgbm_model_local = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    return VotingClassifier(
        estimators=[
            ("cat", cat_model_local),
            ("lgbm", lgbm_model_local),
        ],
        voting="soft",
    )

def build_pipeline(data_node):
    X = data_node.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_node[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = build_voting_classifier()
    return X.skb.apply(vectorizer).skb.apply(model, y=y)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
pred = build_pipeline(data_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).astype(int)
final_validation_score = accuracy_score(valid_part[target_col].astype(int), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
full_pred = build_pipeline(data_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).astype(int)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": np.where(test_pred > 0, True, False),
    }
)
submission.to_csv(SUBMISSION_PATH, index=False)
