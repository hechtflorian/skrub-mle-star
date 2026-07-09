
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

random_state = 42
target_col = "booking_status"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data = skrub.var("data", train_part)
X_train = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)

valid_pred = learner.predict({"data": valid_part})
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")
