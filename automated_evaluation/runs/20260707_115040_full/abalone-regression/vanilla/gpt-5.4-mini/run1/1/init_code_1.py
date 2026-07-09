
import os
import random
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
train_path = "./input/train.csv"
test_path = "./input/test.csv"

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic preprocessing: one-hot encode categorical feature Sex
X = pd.get_dummies(train.drop(columns=["Rings"]))
y = np.log1p(train["Rings"].astype(float))
X_test = pd.get_dummies(test)

# Align train/test columns
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# Hold-out validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

# CatBoostRegressor as specified
model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    verbose=False,
    random_seed=SEED
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

# Validation performance on original Rings scale using RMSLE
valid_pred_log = model.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)
valid_pred = np.clip(valid_pred, 0, None)

valid_true = np.expm1(y_valid)
valid_true = np.clip(valid_true, 0, None)

final_validation_score = np.sqrt(mean_squared_log_error(valid_true, valid_pred))
print(f"Final Validation Performance: {final_validation_score}")

# Train final model on all training data
final_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    verbose=False,
    random_seed=SEED
)
final_model.fit(X, y)

# Predict test set
test_pred_log = final_model.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "Rings": test_pred
})
submission.to_csv("submission.csv", index=False)
