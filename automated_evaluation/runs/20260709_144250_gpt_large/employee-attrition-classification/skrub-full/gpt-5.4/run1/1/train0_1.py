
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "Attrition"

input_dir = Path("./input")
train_path = input_dir / "train.csv"
train_df = pd.read_csv(train_path)

train_part, valid_part = train_test_split(
    train_df,
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgbm_model = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=random_state,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    random_seed=random_state,
    verbose=0,
)

lgbm_predictor = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
cat_predictor = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

lgbm_learner = lgbm_predictor.skb.make_learner(fitted=True)
cat_learner = cat_predictor.skb.make_learner(fitted=True)

valid_pred_lgbm = np.asarray(lgbm_learner.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}), dtype=float).ravel()

valid_pred = 0.5 * valid_pred_lgbm + 0.5 * valid_pred_cat

final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
