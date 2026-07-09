
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

train_df = pd.read_csv("./input/train.csv")

target_col = "yield"
random_state = 42

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=5000,
    loss_function="MAE",
    random_seed=random_state,
    verbose=0
)

pred_graph = X_train.skb.apply(model, y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})

final_validation_score = mean_absolute_error(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
