
import sys
import subprocess
import warnings

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "lightgbm", "-q"])

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_model = CatBoostClassifier(
    verbose=0,
    loss_function="Logloss",
    random_seed=42,
)

lgb_model = LGBMClassifier(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    verbose=-1,
)

cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
lgb_pred = X_train.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train)

cat_learner = cat_pred.skb.make_learner(fitted=True)
lgb_learner = lgb_pred.skb.make_learner(fitted=True)

cat_valid_raw = np.asarray(cat_learner.predict({"data": valid_part})).ravel()
lgb_valid_raw = np.asarray(lgb_learner.predict({"data": valid_part})).ravel()

def to_score_array(pred):
    pred = np.asarray(pred).ravel()
    if pred.dtype == bool:
        return pred.astype(float)
    if np.issubdtype(pred.dtype, np.number):
        pred = pred.astype(float)
        if np.nanmin(pred) < 0.0 or np.nanmax(pred) > 1.0:
            pred_min = np.nanmin(pred)
            pred_max = np.nanmax(pred)
            if pred_max > pred_min:
                pred = (pred - pred_min) / (pred_max - pred_min)
            else:
                pred = np.zeros_like(pred, dtype=float)
        return pred
    return pred.astype(float)

cat_valid_prob = to_score_array(cat_valid_raw)
lgb_valid_prob = to_score_array(lgb_valid_raw)

# Small, fixed soft-vote ensemble with equal weights
ensemble_valid_prob = 0.5 * cat_valid_prob + 0.5 * lgb_valid_prob
ensemble_valid_pred = ensemble_valid_prob >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], ensemble_valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
