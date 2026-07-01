
import subprocess
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Install catboost if not already installed
try:
    import catboost as cb
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "catboost"], check=True)
    import catboost as cb

# Load data
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# Feature engineering: Add rooms per household and bedrooms per room
train['rooms_per_household'] = train['total_rooms'] / train['households']
train['bedrooms_per_room'] = train['total_bedrooms'] / train['total_rooms']
train['population_per_household'] = train['population'] / train['households']

test['rooms_per_household'] = test['total_rooms'] / test['households']
test['bedrooms_per_room'] = test['total_bedrooms'] / test['total_rooms']
test['population_per_household'] = test['population'] / test['households']

# Features and target
X = train.drop('median_house_value', axis=1)
y = train['median_house_value']
X_test = test

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
model = cb.CatBoostRegressor(random_state=42, verbose=0)
model.fit(X_train, y_train)

# Predict on validation set and evaluate
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds) ** 0.5
print(f'Final Validation Performance: {rmse}')

# Predict on test data and save submission
test_preds = model.predict(X_test)
pd.DataFrame({'median_house_value': test_preds}).to_csv('predictions.csv', index=False)
