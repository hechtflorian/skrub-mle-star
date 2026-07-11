
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

random_state = 42
target_col = "Attrition"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data = skrub.var("data", train_part)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()


import numpy as np
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def drop_constants_and_uninformative(df):
    out = df.copy()
    nunique = out.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        out = out.drop(columns=constant_cols)
    return out


data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(drop_constants_and_uninformative)
X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_train[target_col].skb.mark_as_y()

shared_vectorizer = skrub.TableVectorizer()

cat_model = CatBoostClassifier(
    verbose=0,
    random_state=random_state,
    loss_function="Logloss",
    iterations=300,
    depth=6,
    learning_rate=0.08,
)

lgbm_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph_cat = X.skb.apply(shared_vectorizer).skb.apply(cat_model, y=y)
pred_graph_lgbm = X.skb.apply(shared_vectorizer).skb.apply(lgbm_model, y=y)

learner_cat = pred_graph_cat.skb.make_learner(fitted=True)
learner_lgbm = pred_graph_lgbm.skb.make_learner(fitted=True)

valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part})).ravel()
valid_pred_lgbm = np.asarray(learner_lgbm.predict({"data": valid_part})).ravel()

blend_weights = [0.85, 0.9, 0.95]
blend_scores = {}

for w_cat in blend_weights:
    valid_pred = w_cat * valid_pred_cat + (1.0 - w_cat) * valid_pred_lgbm
    blend_scores[w_cat] = roc_auc_score(valid_part[target_col], valid_pred)
    print(f"Ablation[blend_cat_{w_cat:.2f}] roc_auc: {blend_scores[w_cat]}")

best_w_cat = max(blend_scores, key=blend_scores.get)
valid_pred = best_w_cat * valid_pred_cat + (1.0 - best_w_cat) * valid_pred_lgbm
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_df = pd.read_csv("./input/test.csv")

data_full = skrub.var("data", train_df)
data_full = data_full.skb.apply_func(drop_constants_and_uninformative)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_cat = X_full.skb.apply(shared_vectorizer).skb.apply(cat_model, y=y_full)
full_pred_lgbm = X_full.skb.apply(shared_vectorizer).skb.apply(lgbm_model, y=y_full)

full_learner_cat = full_pred_cat.skb.make_learner(fitted=True)
full_learner_lgbm = full_pred_lgbm.skb.make_learner(fitted=True)

test_pred_cat = np.asarray(full_learner_cat.predict({"data": test_df})).ravel()
test_pred_lgbm = np.asarray(full_learner_lgbm.predict({"data": test_df})).ravel()

test_pred = best_w_cat * test_pred_cat + (1.0 - best_w_cat) * test_pred_lgbm

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"EmployeeNumber": test_df["id"], target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)
