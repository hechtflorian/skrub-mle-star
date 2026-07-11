
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "NObeyesdad"

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

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = CatBoostClassifier(
    random_state=random_state,
    verbose=0,
)

predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
