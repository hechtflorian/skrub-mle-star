
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps bind on train_part only for honest holdout
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Minimal fix: impute missing values before vectorization/model fitting
imputer = SimpleImputer(strategy="most_frequent")
vectorizer = skrub.TableVectorizer()

model = LogisticRegression(max_iter=1000)

pred = X_train.skb.apply(imputer).skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

if isinstance(valid_pred, pd.DataFrame):
    valid_pred = valid_pred.iloc[:, 0]
valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool and valid_pred.dtype != np.bool_:
    valid_pred = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
