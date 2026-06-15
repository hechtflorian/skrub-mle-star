
import subprocess
import sys
import json

subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])

import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=skrub.choose_int(80, 220, n_steps=4, default=120, name="n_estimators"),
    learning_rate=skrub.choose_float(0.03, 0.08, log=False, default=0.05, name="learning_rate"),
    verbose=-1,
    n_jobs=1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

search = pred.skb.make_randomized_search(n_iter=4, n_jobs=1, random_state=42, fitted=True)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)

best_params = {}
grid = search.best_params_
if "n_estimators" in grid:
    best_params["n_estimators"] = int(grid["n_estimators"])
elif "data_op__0" in grid:
    best_params["n_estimators"] = int(grid["data_op__0"])
if "learning_rate" in grid:
    best_params["learning_rate"] = float(grid["learning_rate"])
elif "data_op__1" in grid:
    best_params["learning_rate"] = float(grid["data_op__1"])

print(f"Final Validation Performance: {final_validation_score}")
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
