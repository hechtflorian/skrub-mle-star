
import sys
import subprocess
import warnings
from pathlib import Path

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
test_df = pd.read_csv("./input/test.csv")
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

def to_prob(x):
    x = np.asarray(x).ravel()
    if x.dtype == bool:
        return x.astype(float)
    if np.all((x >= 0.0) & (x <= 1.0)):
        return x.astype(float)
    x_min, x_max = np.min(x), np.max(x)
    if np.isclose(x_max, x_min):
        return np.full_like(x, 0.5, dtype=float)
    return (x - x_min) / (x_max - x_min)

cat_valid_prob = to_prob(cat_valid_raw)
lgb_valid_prob = to_prob(lgb_valid_raw)

cat_valid_pred = cat_valid_prob >= 0.5
lgb_valid_pred = lgb_valid_prob >= 0.5

cat_acc = accuracy_score(valid_part[target_col], cat_valid_pred)
lgb_acc = accuracy_score(valid_part[target_col], lgb_valid_pred)

accs = np.array([cat_acc, lgb_acc], dtype=float)
temp = 0.05
weights = np.exp(accs / temp)
weights = weights / weights.sum()
w_cat, w_lgb = weights[0], weights[1]

ensemble_valid_score = w_cat * cat_valid_prob + w_lgb * lgb_valid_prob

both_high = (cat_valid_prob > 0.65) & (lgb_valid_prob > 0.65)
both_low = (cat_valid_prob < 0.35) & (lgb_valid_prob < 0.35)

ensemble_valid_score = np.where(
    both_high,
    np.minimum(1.0, ensemble_valid_score + 0.05),
    ensemble_valid_score,
)
ensemble_valid_score = np.where(
    both_low,
    np.maximum(0.0, ensemble_valid_score - 0.05),
    ensemble_valid_score,
)

thresholds = [0.45, 0.5, 0.55]
best_threshold = 0.5
best_score = -1.0

for thr in thresholds:
    valid_pred = ensemble_valid_score >= thr
    score = accuracy_score(valid_part[target_col], valid_pred)
    if score > best_score:
        best_score = score
        best_threshold = thr

final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

cat_full_pred = X_full.skb.apply(vectorizer).skb.apply(cat_model, y=y_full)
lgb_full_pred = X_full.skb.apply(vectorizer).skb.apply(lgb_model, y=y_full)

cat_full_learner = cat_full_pred.skb.make_learner(fitted=True)
lgb_full_learner = lgb_full_pred.skb.make_learner(fitted=True)

cat_test_raw = np.asarray(cat_full_learner.predict({"data": test_df})).ravel()
lgb_test_raw = np.asarray(lgb_full_learner.predict({"data": test_df})).ravel()

cat_test_prob = to_prob(cat_test_raw)
lgb_test_prob = to_prob(lgb_test_raw)

test_ensemble_score = w_cat * cat_test_prob + w_lgb * lgb_test_prob
test_ensemble_score = np.where(
    (cat_test_prob > 0.65) & (lgb_test_prob > 0.65),
    np.minimum(1.0, test_ensemble_score + 0.05),
    test_ensemble_score,
)
test_ensemble_score = np.where(
    (cat_test_prob < 0.35) & (lgb_test_prob < 0.35),
    np.maximum(0.0, test_ensemble_score - 0.05),
    test_ensemble_score,
)

test_pred = test_ensemble_score >= best_threshold

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred.astype(bool),
    }
)

Path("./final").mkdir(parents=True, exist_ok=True)
submission.to_csv("./final/submission.csv", index=False)
