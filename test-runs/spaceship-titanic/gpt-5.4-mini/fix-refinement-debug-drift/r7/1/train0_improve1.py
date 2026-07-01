
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prefer native package imports; fall back only if needed.
try:
    from catboost import CatBoostClassifier
except Exception:
    from sklearn.ensemble import RandomForestClassifier as CatBoostClassifier  # fallback only if needed

try:
    from lightgbm import LGBMClassifier
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier  # fallback only if needed

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"

# Smallest safe fix: make boolean-like columns uniformly string-typed
# so TableVectorizer's ordinal encoding path never sees mixed bool/str.
for col in ["CryoSleep", "VIP"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype("string")
    if col in test_df.columns:
        test_df[col] = test_df[col].astype("string")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

low_card_pipe = skrub.TableVectorizer()
high_card_pipe = skrub.TableVectorizer()

catboost_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=300,
    depth=6,
    learning_rate=0.05,
    verbose=0,
    random_seed=42,
)

pred_cat = X_train.skb.apply(low_card_pipe).skb.apply(catboost_model, y=y_train)
learner_cat = pred_cat.skb.make_learner(fitted=True)
valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part})).ravel()

lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbose=-1,
)

pred_lgbm = X_train.skb.apply(high_card_pipe).skb.apply(lgbm_model, y=y_train)
learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
valid_pred_lgbm = np.asarray(learner_lgbm.predict({"data": valid_part})).ravel()

def _to_bool_preds(preds):
    preds = np.asarray(preds)
    if preds.dtype == bool:
        return preds
    if preds.ndim > 1:
        preds = preds[:, -1]
    if preds.dtype.kind in "fc":
        return preds >= 0.5
    return preds.astype(str) == "True"

valid_bool_cat = _to_bool_preds(valid_pred_cat)
valid_bool_lgbm = _to_bool_preds(valid_pred_lgbm)
y_valid = valid_part[target_col].astype(bool).to_numpy()

score_cat = accuracy_score(y_valid, valid_bool_cat)
score_lgbm = accuracy_score(y_valid, valid_bool_lgbm)

if score_lgbm >= score_cat:
    final_validation_score = score_lgbm
    best_model = lgbm_model
    best_pipe = high_card_pipe
else:
    final_validation_score = score_cat
    best_model = catboost_model
    best_pipe = low_card_pipe

print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(best_pipe).skb.apply(best_model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = np.asarray(full_learner.predict({"data": test_df})).ravel()

if test_pred.dtype != bool:
    if test_pred.ndim > 1:
        test_pred = test_pred[:, -1]
    if test_pred.dtype.kind in "fc":
        test_pred = test_pred >= 0.5
    else:
        test_pred = test_pred.astype(str) == "True"

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred.astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
