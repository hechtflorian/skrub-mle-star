
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

random_state = 42
target_col = "count"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_df["datetime"] = pd.to_datetime(train_df["datetime"])
test_df["datetime"] = pd.to_datetime(test_df["datetime"])

train_df = train_df.sort_values("datetime").reset_index(drop=True)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})
valid_pred = np.maximum(valid_pred, 0)
final_validation_score = mean_squared_log_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
