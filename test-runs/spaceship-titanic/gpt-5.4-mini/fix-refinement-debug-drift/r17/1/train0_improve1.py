
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import os

random_state = 42
test_size = 0.2
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = HistGradientBoostingClassifier(random_state=random_state)
pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

valid_pred = np.asarray(valid_pred)
if valid_pred.ndim > 1:
    valid_pred = valid_pred.ravel()
if valid_pred.dtype != bool:
    valid_pred = valid_pred.astype(int) if np.issubdtype(valid_pred.dtype, np.number) else valid_pred

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
