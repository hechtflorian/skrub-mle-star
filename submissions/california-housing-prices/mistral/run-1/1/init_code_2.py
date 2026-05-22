
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Features and target
X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]
X_test = test.copy()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)

# Predict on validation set
val_preds = model.predict(X_val)

# Calculate RMSE
rmse = mean_squared_error(y_val, val_preds) ** 0.5

# Print final performance
print(f'Final Validation Performance: {rmse}')

# Predict on test set and save predictions
test_preds = model.predict(X_test)
submission = pd.DataFrame({"median_house_value": test_preds})
submission.to_csv("submission.csv", index=False, header=True)
