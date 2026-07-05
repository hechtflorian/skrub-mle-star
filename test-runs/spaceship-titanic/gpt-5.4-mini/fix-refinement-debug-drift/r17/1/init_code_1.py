
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "Transported"
random_state = 42

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding must use train_part only for honest holdout validation
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# DataOps pipeline
vectorizer = skrub.TableVectorizer()
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.08,
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

# Fit on train_part only, then predict on valid_part without target leakage/schema mismatch
val_learner = pred.skb.make_learner(fitted=True)
valid_features = valid_part.drop(columns=target_col, errors="ignore")
valid_pred = val_learner.predict({"data": valid_features})

# Ensure boolean predictions for accuracy_score
if valid_pred.dtype != bool:
    if np.issubdtype(valid_pred.dtype, np.floating):
        valid_pred = valid_pred >= 0.5
    else:
        valid_pred = pd.Series(valid_pred).astype(str).str.lower().isin(["true", "1", "yes"])

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
