
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

random_state = 0
target_col = "revenue"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

if "Open Date" in train_df.columns:
    train_df["Open Date"] = pd.to_datetime(train_df["Open Date"], errors="coerce")
if "Open Date" in test_df.columns:
    test_df["Open Date"] = pd.to_datetime(test_df["Open Date"], errors="coerce")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
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

full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(
    CatBoostRegressor(
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
    ),
    y=y_full,
)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
submission = pd.DataFrame({"Id": test_df["Id"], "Prediction": test_pred})
submission.to_csv("./submission.csv", index=False)
