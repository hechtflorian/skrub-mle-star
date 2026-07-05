
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

def feature_engineer(df):
    df = df.copy()
    df["Cabin"] = df["Cabin"].fillna("Unknown/Unknown/Unknown")
    df["Name"] = df["Name"].fillna("Unknown")
    return df

train_df = feature_engineer(train_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.StringEncoder(),
)

model = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool:
    if valid_pred.ndim > 1 and valid_pred.shape[1] == 2:
        valid_pred = valid_pred[:, 1]
    valid_pred = (valid_pred > 0.5)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
