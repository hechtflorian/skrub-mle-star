
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
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

vectorizer = skrub.TableVectorizer()
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1200,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    random_seed=random_state,
    verbose=0,
)

predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
