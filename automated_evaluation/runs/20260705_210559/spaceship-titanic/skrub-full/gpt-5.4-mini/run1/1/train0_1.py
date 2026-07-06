
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state, shuffle=True
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_cat = skrub.TableVectorizer()

# Minimal fix: keep the DataOps pipeline, but ensure the final estimator is a
# proper sklearn classifier with predictable predict_proba support.
# The reported issue was around the fit/predict flow with skrub wrapping.
cat_model = RandomForestClassifier(
    n_estimators=300,
    random_state=random_state,
    n_jobs=-1,
)

pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)
learner_cat = pred_cat.skb.make_learner(fitted=True)

valid_pred_proba = learner_cat.predict_proba({"data": valid_part})
valid_pred = np.asarray(valid_pred_proba)[:, 1] >= 0.5
valid_pred = pd.Series(valid_pred, index=valid_part.index).map({True: True, False: False})

final_validation_score = accuracy_score(valid_part[target_col].astype(bool), valid_pred.astype(bool))
print(f"Final Validation Performance: {final_validation_score}")
