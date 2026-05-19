
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_df = pd.read_csv('./input/train.csv')

# Define the features and target
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor model
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = gbr_model.predict(X_val)

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Print the final validation performance
print(f'Final Validation Performance: {rmse}')
