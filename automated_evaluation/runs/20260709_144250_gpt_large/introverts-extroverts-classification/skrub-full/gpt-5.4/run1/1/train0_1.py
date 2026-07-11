
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "Personality"

train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)

X_train_cat = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_cat = data_train[target_col].skb.mark_as_y()

X_train_lgb = data_train.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
y_train_lgb = (
    data_train[target_col].replace({"Introvert": 0, "Extrovert": 1}).skb.mark_as_y()
)

vectorizer_cat = skrub.TableVectorizer()
vectorizer_lgb = skrub.TableVectorizer()

predictor_cat = X_train_cat.skb.apply(vectorizer_cat).skb.apply(
    CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    ),
    y=y_train_cat,
)

predictor_lgb = X_train_lgb.skb.apply(vectorizer_lgb).skb.apply(
    LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    ),
    y=y_train_lgb,
)

learner_cat = predictor_cat.skb.make_learner(fitted=True)
learner_lgb = predictor_lgb.skb.make_learner(fitted=True)

valid_pred_cat = learner_cat.predict({"data": valid_part})
valid_pred_lgb_num = learner_lgb.predict({"data": valid_part})
valid_pred_lgb = (
    pd.Series(valid_pred_lgb_num).map({0: "Introvert", 1: "Extrovert"}).to_numpy()
)

valid_pred_cat_num = pd.Series(valid_pred_cat).map({"Introvert": 0, "Extrovert": 1}).to_numpy()
valid_blend_num = ((valid_pred_cat_num + valid_pred_lgb_num) >= 1).astype(int)
valid_pred = pd.Series(valid_blend_num).map({0: "Introvert", 1: "Extrovert"}).to_numpy()

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
