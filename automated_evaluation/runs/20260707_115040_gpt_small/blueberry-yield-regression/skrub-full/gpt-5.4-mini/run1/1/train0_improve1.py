
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

random_state = 42
test_size = 0.2
target_col = "yield"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = CatBoostRegressor(
    loss_function="MAE",
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred.skb.make_learner(fitted=True)
valid_pred = np.asarray(learner.predict({"data": valid_part}), dtype=float).ravel()
final_validation_score = mean_absolute_error(valid_part[target_col].to_numpy(), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
