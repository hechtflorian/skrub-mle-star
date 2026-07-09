
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "yield"
random_state = 42
test_size = 0.2

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=random_state,
    n_jobs=1,
    verbosity=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_absolute_error(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()
final_pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
final_learner = final_pred_graph.skb.make_learner(fitted=True)
test_pred = final_learner.predict({"data": test_df})

submission = pd.DataFrame({"id": test_df["id"], "yield": np.asarray(test_pred)})
submission.to_csv("submission.csv", index=False)
