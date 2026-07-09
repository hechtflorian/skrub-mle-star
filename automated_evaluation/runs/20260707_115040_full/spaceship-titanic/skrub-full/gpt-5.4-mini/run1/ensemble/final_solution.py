
import os
import pandas as pd
import numpy as np
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = RandomForestClassifier(
    n_estimators=300,
    random_state=random_state,
    n_jobs=-1,
)

predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_df = pd.read_csv("./input/test.csv")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)
