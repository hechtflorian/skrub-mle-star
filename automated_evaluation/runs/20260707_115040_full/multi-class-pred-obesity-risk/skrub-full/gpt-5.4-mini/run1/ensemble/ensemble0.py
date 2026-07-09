
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

random_state = 42
test_size = 0.2
target_col = "NObeyesdad"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=test_size,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgbm_model = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=random_state,
    n_jobs=1,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=400,
    learning_rate=0.1,
    depth=6,
    random_seed=random_state,
    verbose=0,
)

pred_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

def _get_pred_and_proba(learner, df):
    pred = np.asarray(learner.predict({"data": df}))
    proba = None
    try:
        proba = np.asarray(learner.predict_proba({"data": df}))
    except Exception:
        proba = None
    return pred, proba

valid_pred_lgbm, valid_proba_lgbm = _get_pred_and_proba(learner_lgbm, valid_part)
valid_pred_cat, valid_proba_cat = _get_pred_and_proba(learner_cat, valid_part)

if valid_pred_lgbm.ndim > 1:
    valid_pred_lgbm = valid_pred_lgbm.argmax(axis=1)
if valid_pred_cat.ndim > 1:
    valid_pred_cat = valid_pred_cat.argmax(axis=1)

if valid_proba_lgbm is not None and valid_proba_cat is not None:
    if valid_proba_lgbm.ndim == 1:
        valid_proba_lgbm = valid_proba_lgbm.reshape(-1, 1)
    if valid_proba_cat.ndim == 1:
        valid_proba_cat = valid_proba_cat.reshape(-1, 1)

    if valid_proba_lgbm.shape == valid_proba_cat.shape:
        blended_proba = 0.4 * valid_proba_lgbm + 0.6 * valid_proba_cat
        valid_pred = blended_proba.argmax(axis=1)
    else:
        valid_pred = []
        for i in range(len(valid_part)):
            if valid_pred_lgbm[i] == valid_pred_cat[i]:
                valid_pred.append(valid_pred_cat[i])
            else:
                cat_conf = float(np.max(valid_proba_cat[i])) if valid_proba_cat.ndim == 2 else float(valid_proba_cat[i])
                lgbm_conf = float(np.max(valid_proba_lgbm[i])) if valid_proba_lgbm.ndim == 2 else float(valid_proba_lgbm[i])
                valid_pred.append(valid_pred_cat[i] if cat_conf >= lgbm_conf else valid_pred_lgbm[i])
        valid_pred = np.asarray(valid_pred)
else:
    valid_pred = []
    for p_lgbm, p_cat in zip(valid_pred_lgbm, valid_pred_cat):
        if p_lgbm == p_cat:
            valid_pred.append(p_cat)
        else:
            valid_pred.append(p_cat)
    valid_pred = np.asarray(valid_pred)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
