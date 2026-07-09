
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

random_state = 42
target_col = "Attrition"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data = skrub.var("data", train_part)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph = X.skb.apply(vectorizer).skb.apply(model, y=y)
learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})

final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
