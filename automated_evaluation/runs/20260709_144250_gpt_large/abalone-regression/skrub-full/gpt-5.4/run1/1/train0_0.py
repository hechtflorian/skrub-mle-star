
import os
import glob
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

random_state = 42
target_col = "Rings"

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5

def find_csv(name):
    candidates = [
        os.path.join(".", "input", name),
        os.path.join(".", "input", "*", name),
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under ./input")

train_path = find_csv("train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=random_state,
    verbose=-1
)

predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.clip(np.asarray(valid_pred), 0, None)

final_validation_score = rmsle(valid_part[target_col].to_numpy(), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
