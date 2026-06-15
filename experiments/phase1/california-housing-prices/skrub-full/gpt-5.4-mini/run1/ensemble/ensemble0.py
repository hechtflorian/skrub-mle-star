
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Fix: catboost is unavailable in this environment, so use a sklearn-compatible fallback
# while keeping the same model family slot as a tree-based boosting regressor.
try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    from sklearn.ensemble import HistGradientBoostingRegressor as CatBoostRegressor

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

# Keep CatBoost-style parameters when available; fallback ignores unsupported params safely.
try:
    model1 = CatBoostRegressor(
        loss_function="RMSE",
        verbose=0,
        random_seed=42,
        iterations=500,
        learning_rate=0.05,
        depth=8,
    )
except TypeError:
    model1 = CatBoostRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        random_state=42,
    )

# Thin ensemble member 1: original pipeline as-is
pred1 = X_train.skb.apply(vectorizer).skb.apply(model1, y=y_train)
learner1 = pred1.skb.make_learner(fitted=True)
valid_pred1 = learner1.predict({"data": valid_part})

# Thin ensemble member 2: structurally similar booster with a slightly different capacity
try:
    model2 = CatBoostRegressor(
        loss_function="RMSE",
        verbose=0,
        random_seed=7,
        iterations=700,
        learning_rate=0.03,
        depth=6,
    )
except TypeError:
    model2 = CatBoostRegressor(
        learning_rate=0.03,
        max_depth=6,
        max_iter=700,
        random_state=7,
    )

pred2 = X_train.skb.apply(vectorizer).skb.apply(model2, y=y_train)
learner2 = pred2.skb.make_learner(fitted=True)
valid_pred2 = learner2.predict({"data": valid_part})

# Tiny validation-tuned scalar blend over [0, 1]
best_w = 0.5
best_rmse = float("inf")
for w in np.linspace(0.0, 1.0, 21):
    ensemble_valid_pred = w * valid_pred1 + (1.0 - w) * valid_pred2
    rmse = mean_squared_error(valid_part[target_col], ensemble_valid_pred) ** 0.5
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

# Final validation performance from the tuned blend
final_validation_score = best_rmse
print(f"Final Validation Performance: {final_validation_score}")

# Apply the same weighted average to test predictions from the two fitted learners
test_pred1 = learner1.predict({"data": test_df})
test_pred2 = learner2.predict({"data": test_df})
test_pred = best_w * test_pred1 + (1.0 - best_w) * test_pred2

# Keep submission/export behavior minimal and non-invasive
submission = pd.DataFrame({
    "id": test_df["id"] if "id" in test_df.columns else np.arange(len(test_df)),
    target_col: test_pred
})
submission.to_csv("submission.csv", index=False)
