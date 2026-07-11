
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

# Load data
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "yield"
random_state = 42

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph bound to train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Model
cat_model = CatBoostRegressor(
    loss_function="MAE",
    verbose=0,
    random_seed=random_state,
)

pred = X_train.skb.apply(
    skrub.TableVectorizer(),
).skb.apply(cat_model, y=y_train)

# Fit on train_part and evaluate on valid_part
val_learner = pred.skb.make_learner(fitted=True)
cat_valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = mean_absolute_error(valid_part[target_col], cat_valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
