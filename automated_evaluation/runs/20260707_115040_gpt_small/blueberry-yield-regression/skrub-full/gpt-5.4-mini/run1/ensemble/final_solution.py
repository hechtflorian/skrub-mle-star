import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import os

random_state = 42
test_size = 0.2
target_col = "yield"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# Shared upstream DataOps root
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Leg 1: original pipeline
vectorizer_1 = skrub.TableVectorizer()
model_1 = CatBoostRegressor(
    loss_function="MAE",
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=random_state,
    verbose=0,
)

pred_1 = X_train.skb.apply(vectorizer_1).skb.apply(model_1, y=y_train)
learner_1 = pred_1.skb.make_learner(fitted=True)

# Leg 2: light variant, same DataOps pattern with a slightly different vectorizer/model view
vectorizer_2 = skrub.TableVectorizer()
model_2 = CatBoostRegressor(
    loss_function="MAE",
    iterations=1200,
    learning_rate=0.04,
    depth=5,
    random_seed=random_state + 7,
    verbose=0,
)

pred_2 = X_train.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train)
learner_2 = pred_2.skb.make_learner(fitted=True)

valid_pred_1 = np.asarray(learner_1.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_2 = np.asarray(learner_2.predict({"data": valid_part}), dtype=float).ravel()

# Minimal deterministic blend
valid_pred = 0.6 * valid_pred_1 + 0.4 * valid_pred_2

final_validation_score = mean_absolute_error(valid_part[target_col].to_numpy(), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_df = pd.read_csv("./input/test.csv")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_1 = X_full.skb.apply(vectorizer_1).skb.apply(model_1, y=y_full)
full_learner_1 = full_pred_1.skb.make_learner(fitted=True)

full_pred_2 = X_full.skb.apply(vectorizer_2).skb.apply(model_2, y=y_full)
full_learner_2 = full_pred_2.skb.make_learner(fitted=True)

test_pred_1 = np.asarray(full_learner_1.predict({"data": test_df}), dtype=float).ravel()
test_pred_2 = np.asarray(full_learner_2.predict({"data": test_df}), dtype=float).ravel()

test_pred = 0.6 * test_pred_1 + 0.4 * test_pred_2

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"id": test_df["id"], target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)
