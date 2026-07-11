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

valid_pred_lgbm = learner_lgbm.predict({"data": valid_part})
valid_pred_cat = learner_cat.predict({"data": valid_part})

valid_pred_lgbm = np.asarray(valid_pred_lgbm)
valid_pred_cat = np.asarray(valid_pred_cat)

if valid_pred_lgbm.ndim > 1:
    valid_pred_lgbm = valid_pred_lgbm.argmax(axis=1)
if valid_pred_cat.ndim > 1:
    valid_pred_cat = valid_pred_cat.argmax(axis=1)

valid_pred = []
for p1, p2 in zip(valid_pred_lgbm, valid_pred_cat):
    valid_pred.append(p1 if p1 == p2 else p1)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_path = os.path.join("./input", "test.csv")
test_df = pd.read_csv(test_path)

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_lgbm = X_full.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_full)
full_pred_cat = X_full.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_full)

full_learner_lgbm = full_pred_lgbm.skb.make_learner(fitted=True)
full_learner_cat = full_pred_cat.skb.make_learner(fitted=True)

test_pred_lgbm = full_learner_lgbm.predict({"data": test_df})
test_pred_cat = full_learner_cat.predict({"data": test_df})

test_pred_lgbm = np.asarray(test_pred_lgbm)
test_pred_cat = np.asarray(test_pred_cat)

if test_pred_lgbm.ndim > 1:
    test_pred_lgbm = test_pred_lgbm.argmax(axis=1)
if test_pred_cat.ndim > 1:
    test_pred_cat = test_pred_cat.argmax(axis=1)

test_pred = []
for p1, p2 in zip(test_pred_lgbm, test_pred_cat):
    test_pred.append(p1 if p1 == p2 else p1)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"id": test_df["id"], target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)