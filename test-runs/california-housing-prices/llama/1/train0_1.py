
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
train_df = pd.read_csv('./input/train.csv')

# Define the features and target
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# Initialize and fit the Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the validation set using both models
gbr_pred = gbr.predict(X_val)
rf_pred = rf.predict(X_val)

# Ensemble the predictions
y_pred = (gbr_pred + rf_pred) / 2

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f'Final Validation Performance: {rmse}')
