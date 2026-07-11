
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
valid_pred_1 = np.asarray(learner_1.predict({"data": valid_part}), dtype=float).ravel()

vectorizer_2 = skrub.TableVectorizer()
model_2 = CatBoostRegressor(
    loss_function="RMSE",
    iterations=900,
    learning_rate=0.04,
    depth=7,
    random_seed=random_state + 7,
    verbose=0,
)

pred_2 = X_train.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train)
learner_2 = pred_2.skb.make_learner(fitted=True)
valid_pred_2 = np.asarray(learner_2.predict({"data": valid_part}), dtype=float).ravel()

linear_blend = 0.65 * valid_pred_1 + 0.35 * valid_pred_2

r1 = pd.Series(valid_pred_1).rank(pct=True).to_numpy()
r2 = pd.Series(valid_pred_2).rank(pct=True).to_numpy()
blend_rank = 0.65 * r1 + 0.35 * r2

sorted_y = np.sort(train_part[target_col].to_numpy(dtype=float))
n = len(sorted_y)
quantiles = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])

blend_rank_clip = np.clip(blend_rank, 0.0, 1.0)
quantile_blend = np.interp(blend_rank_clip, quantiles, sorted_y)

valid_pred = 0.5 * linear_blend + 0.5 * quantile_blend

final_validation_score = mean_absolute_error(valid_part[target_col].to_numpy(), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
