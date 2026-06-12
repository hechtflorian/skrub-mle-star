
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")

# Load data
train_df = pd.read_csv(train_path)

target_col = "median_house_value"

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Fixed CatBoost model baked from best params:
# model_variant = 2 -> d8_lr0.03
vectorizer = skrub.TableVectorizer()
model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    random_seed=42,
    verbose=0,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
