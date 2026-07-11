
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "booking_status"

train_df = pd.read_csv("./input/train.csv")

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

vectorizer_cat = skrub.TableVectorizer()
vectorizer_lgb = skrub.TableVectorizer()

cat_predictor = X_train.skb.apply(vectorizer_cat).skb.apply(
    CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    ),
    y=y_train,
)

lgb_predictor = X_train.skb.apply(vectorizer_lgb).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    ),
    y=y_train,
)

cat_learner = cat_predictor.skb.make_learner(fitted=True)
lgb_learner = lgb_predictor.skb.make_learner(fitted=True)

cat_valid_pred = cat_learner.predict_proba({"data": valid_part})[:, 1]
lgb_valid_pred = lgb_learner.predict_proba({"data": valid_part})[:, 1]

valid_pred = 0.5 * cat_valid_pred + 0.5 * lgb_valid_pred
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")
