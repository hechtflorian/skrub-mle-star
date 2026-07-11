
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
TARGET_COL = "Attrition"
ID_COL = "id"

train_path = "./input/train.csv"

train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_df[TARGET_COL],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_train = data_train[TARGET_COL].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_X_train = X_train.skb.apply(vectorizer)
cat_model = CatBoostClassifier(
    random_state=RANDOM_STATE,
    verbose=0,
    cat_features=None,
)
cat_predictor = cat_X_train.skb.apply(cat_model, y=y_train)

lgbm_X_train = X_train.skb.apply(vectorizer)
lgbm_model = LGBMClassifier(
    random_state=RANDOM_STATE,
    verbose=-1,
    n_estimators=100,
    num_leaves=61,
    learning_rate=0.05042162933217116,
)
lgbm_predictor = lgbm_X_train.skb.apply(lgbm_model, y=y_train)

cat_learner = cat_predictor.skb.make_learner(fitted=True)
lgbm_learner = lgbm_predictor.skb.make_learner(fitted=True)

cat_valid_pred = np.asarray(cat_learner.predict({"data": valid_part}))
lgbm_valid_pred = np.asarray(lgbm_learner.predict({"data": valid_part}))

if set(np.unique(cat_valid_pred)).issubset({0, 1}):
    cat_valid_pred = cat_learner.predict_proba({"data": valid_part})[:, 1]
if set(np.unique(lgbm_valid_pred)).issubset({0, 1}):
    lgbm_valid_pred = lgbm_learner.predict_proba({"data": valid_part})[:, 1]

ensemble_valid_pred = 0.5 * np.asarray(cat_valid_pred) + 0.5 * np.asarray(lgbm_valid_pred)
final_validation_score = roc_auc_score(valid_part[TARGET_COL], ensemble_valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
