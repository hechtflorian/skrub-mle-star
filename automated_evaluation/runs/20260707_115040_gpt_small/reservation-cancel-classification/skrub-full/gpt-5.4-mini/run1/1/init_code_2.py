
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
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
X_train = data.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
y_train = data[target_col].skb.mark_as_y()

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=random_state,
    verbose=0,
    allow_writing_files=False,
)

pred_graph = X_train.skb.apply(model, y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)

valid_X = valid_part.drop(columns=[target_col, "id"], errors="ignore")
valid_pred = learner.predict({"data": valid_part})

final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
