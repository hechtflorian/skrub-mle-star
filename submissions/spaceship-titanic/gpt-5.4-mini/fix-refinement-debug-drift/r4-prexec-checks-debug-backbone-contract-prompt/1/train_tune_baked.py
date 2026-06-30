
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")

target_col = "Transported"

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Minimal fix: vectorize categorical/string columns before CatBoost
# so CatBoost receives only numeric features.
vectorizer = skrub.TableVectorizer()

pred = X_train.skb.apply(vectorizer).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_train,
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_pred_bool = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_bool)
print(f"Final Validation Performance: {final_validation_score}")
