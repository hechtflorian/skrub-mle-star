
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "Transported"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Small feature engineering, keeping the existing DataOps structure intact.
def add_features(df):
    df = df.copy()

    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string")
        parts = cabin.str.split("/", expand=True)
        if parts.shape[1] >= 3:
            df["CabinDeck"] = parts[0]
            df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
            df["CabinSide"] = parts[2]
        else:
            df["CabinDeck"] = pd.NA
            df["CabinNum"] = pd.NA
            df["CabinSide"] = pd.NA

    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype("string").str.len()
        df["Surname"] = df["Name"].astype("string").str.split().str[-1]
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[TARGET_COL]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_train = data_train[TARGET_COL].skb.mark_as_y()

# Minimal fix for the reported CatBoost dtype mismatch:
# TableVectorizer can preserve pandas category dtypes, so explicitly tell CatBoost
# those columns are categorical.
vectorizer = skrub.TableVectorizer()

def _catboost_cat_features_from_df(df):
    return [c for c in df.columns if str(df[c].dtype) in ("category", "object", "string")]

cat_params = dict(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
)

# Build the pipeline in DataOps style; keep the main workflow intact.
# The smallest practical fix is to provide cat_features to CatBoost.
# Since the vectorizer output may still contain category columns, compute them
# from the transformed training frame at fit time.
class CatBoostWithAutoCats(CatBoostClassifier):
    def fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame) and "cat_features" not in kwargs:
            kwargs["cat_features"] = _catboost_cat_features_from_df(X)
        return super().fit(X, y=y, **kwargs)

cat_model = CatBoostWithAutoCats(**cat_params)
lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

# Keep both required estimators in the structural solution.
# A simple soft-vote ensemble is used over the two existing model families.
def build_voting_classifier():
    return VotingClassifier(
        estimators=[
            ("cat", cat_model),
            ("lgbm", lgbm_model),
        ],
        voting="soft",
        weights=[2, 1],
    )

pred = X_train.skb.apply(vectorizer).skb.apply(build_voting_classifier(), y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)
valid_pred = (valid_pred >= 0.5).astype(bool) if valid_pred.dtype != bool else valid_pred

final_validation_score = accuracy_score(valid_part[TARGET_COL], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: refit on full training data and predict test.
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_full = data_full[TARGET_COL].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(build_voting_classifier(), y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred)
test_pred = (test_pred >= 0.5).astype(bool) if test_pred.dtype != bool else test_pred

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
