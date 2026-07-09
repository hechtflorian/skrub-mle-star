
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

random_state = 42
target_col = "revenue"

train_path = glob.glob(os.path.join("./input", "train.csv"))[0]
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    low_cardinality="drop",
    high_cardinality="drop",
)

model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
