
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

target_col = "Transported"
random_state = 42

train_df = pd.read_csv(TRAIN_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
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
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)

predictor_lgbm = (
    X_train
    .skb.apply(vectorizer_lgbm)
    .skb.apply(lgbm_model, y=y_train)
)

predictor_cat = (
    X_train
    .skb.apply(vectorizer_cat)
    .skb.apply(cat_model, y=y_train)
)

learner_lgbm = predictor_lgbm.skb.make_learner(fitted=True)
learner_cat = predictor_cat.skb.make_learner(fitted=True)

valid_pred_lgbm = pd.Series(learner_lgbm.predict({"data": valid_part}), index=valid_part.index)
valid_pred_cat = pd.Series(learner_cat.predict({"data": valid_part}), index=valid_part.index)

valid_true = pd.Series(valid_part[target_col], index=valid_part.index).astype(str).map({"True": True, "False": False}).fillna(valid_part[target_col])

valid_pred_lgbm_bool = valid_pred_lgbm.astype(str).map({"True": True, "False": False}).fillna(valid_pred_lgbm)
valid_pred_cat_bool = valid_pred_cat.astype(str).map({"True": True, "False": False}).fillna(valid_pred_cat)

valid_pred_lgbm_num = pd.Series(valid_pred_lgbm_bool, index=valid_part.index).astype(bool).astype(int)
valid_pred_cat_num = pd.Series(valid_pred_cat_bool, index=valid_part.index).astype(bool).astype(int)

ensemble_vote = ((valid_pred_lgbm_num + valid_pred_cat_num) >= 1).astype(bool)

final_validation_score = accuracy_score(valid_true.astype(bool), ensemble_vote)
print(f"Final Validation Performance: {final_validation_score}")
