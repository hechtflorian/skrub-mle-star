
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import HistGradientBoostingRegressor

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

# Base model: CatBoost-style fallback slot
try:
    model_base = CatBoostRegressor(
        loss_function="RMSE",
        verbose=0,
        random_seed=42,
        iterations=500,
        learning_rate=0.05,
        depth=8,
    )
except NameError:
    from sklearn.ensemble import HistGradientBoostingRegressor as CatBoostRegressor

    model_base = CatBoostRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        random_state=42,
    )
except TypeError:
    model_base = CatBoostRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        random_state=42,
    )

# Reference model integrated as an additional learner
model_ref = HistGradientBoostingRegressor(random_state=42)

pred_base = X_train.skb.apply(vectorizer).skb.apply(model_base, y=y_train)
pred_ref = X_train.skb.apply(vectorizer).skb.apply(model_ref, y=y_train)

val_learner_base = pred_base.skb.make_learner(fitted=True)
val_learner_ref = pred_ref.skb.make_learner(fitted=True)

valid_pred_base = np.asarray(val_learner_base.predict({"data": valid_part}))
valid_pred_ref = np.asarray(val_learner_ref.predict({"data": valid_part}))

# Simple ensemble of both model families
valid_pred = 0.5 * valid_pred_base + 0.5 * valid_pred_ref

final_validation_score = mean_squared_error(
    valid_part[target_col], valid_pred
) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")
