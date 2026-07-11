
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

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgbm_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    verbose=0,
    random_state=random_state,
    loss_function="Logloss",
    iterations=300,
    depth=6,
    learning_rate=0.08,
)

pred_graph_lgbm = X.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y)
pred_graph_cat = X.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y)

learner_lgbm = pred_graph_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_graph_cat.skb.make_learner(fitted=True)

valid_pred_lgbm = np.asarray(learner_lgbm.predict({"data": valid_part})).ravel()
valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part})).ravel()

valid_pred = 0.6 * valid_pred_lgbm + 0.4 * valid_pred_cat
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
