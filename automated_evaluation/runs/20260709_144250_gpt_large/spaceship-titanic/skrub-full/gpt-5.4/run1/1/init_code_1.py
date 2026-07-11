
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

target_col = "Transported"
random_state = 42

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

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

feature_builder = skrub.TableVectorizer()

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)

predictor = (
    X_train
    .skb.apply(feature_builder)
    .skb.apply(
        model,
        y=y_train,
    )
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

if pd.api.types.is_bool_dtype(valid_part[target_col]):
    valid_true = valid_part[target_col]
else:
    valid_true = valid_part[target_col].astype(str).map({"True": True, "False": False})

if not isinstance(valid_pred, (pd.Series, np.ndarray, list)):
    valid_pred = np.asarray(valid_pred)
valid_pred = pd.Series(valid_pred).astype(str).map({"True": True, "False": False}).fillna(pd.Series(valid_pred))

final_validation_score = accuracy_score(valid_true, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
