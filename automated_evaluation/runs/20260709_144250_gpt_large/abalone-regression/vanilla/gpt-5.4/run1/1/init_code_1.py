
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

DATA_DIR = "./input"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

features = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Whole weight.1",
    "Whole weight.2",
    "Shell weight",
]
cat_features = ["Sex"]
target_col = "Rings"

X = train[features].copy()
y = train[target_col].clip(lower=0)
X_test = test[features].copy()

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=200,
)

model.fit(
    X_tr,
    y_tr,
    cat_features=cat_features,
    eval_set=(X_va, y_va),
    use_best_model=True,
    early_stopping_rounds=200,
)

val_pred = model.predict(X_va)
val_pred = np.clip(val_pred, 0, None)

rmsle = np.sqrt(mean_squared_log_error(y_va, val_pred))

test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "id": test["id"],
    "Rings": test_pred
})
submission.to_csv("submission_catboost.csv", index=False)

print(f"Final Validation Performance: {rmsle}")
