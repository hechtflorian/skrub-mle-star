
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import skrub
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

random_state = 42
target_col = "booking_status"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data = skrub.var("data", train_part)
X_train = data.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
y_train = data[target_col].skb.mark_as_y()

# Model 1: LightGBM baseline from base solution
vectorizer = skrub.TableVectorizer()
model_lgb = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph_lgb = X_train.skb.apply(vectorizer).skb.apply(model_lgb, y=y_train)
learner_lgb = pred_graph_lgb.skb.make_learner(fitted=True)

# Model 2: CatBoost reference solution
model_cat = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=random_state,
    verbose=0,
    allow_writing_files=False,
)

pred_graph_cat = X_train.skb.apply(model_cat, y=y_train)
learner_cat = pred_graph_cat.skb.make_learner(fitted=True)

valid_pred_lgb = np.asarray(learner_lgb.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part}), dtype=float).ravel()

meta_X_valid = np.column_stack([
    valid_pred_lgb,
    valid_pred_cat,
    np.abs(valid_pred_lgb - valid_pred_cat),
])

meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
meta_learner.fit(meta_X_valid, valid_part[target_col].values)

valid_pred = meta_learner.predict_proba(meta_X_valid)[:, 1]
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")
