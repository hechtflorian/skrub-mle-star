
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TARGET = "median_house_value"
RANDOM_STATE = 42

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# Block 1: honest holdout validation
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_train = data_train[TARGET].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    low_cardinality="passthrough",
    high_cardinality=skrub.StringEncoder(n_components=20),
)

model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.03,
    iterations=5000,
    random_seed=RANDOM_STATE,
    verbose=200,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred_graph.skb.make_learner(fitted=True)

valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[TARGET], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Block 2: full train for test predictions
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_full = data_full[TARGET].skb.mark_as_y()

full_pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred_graph.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
