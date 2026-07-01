
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

print(f'Original Model Performance: {rmse}')

# Ablation 1: Reduce the number of estimators
gbr_model_ablation_1 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
gbr_model_ablation_1.fit(X_train, y_train)
y_pred_ablation_1 = gbr_model_ablation_1.predict(X_val)
rmse_ablation_1 = np.sqrt(mean_squared_error(y_val, y_pred_ablation_1))
print(f'Ablation 1 Performance (Reduced Estimators): {rmse_ablation_1}')

# Ablation 2: Increase the learning rate
gbr_model_ablation_2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, random_state=42)
gbr_model_ablation_2.fit(X_train, y_train)
y_pred_ablation_2 = gbr_model_ablation_2.predict(X_val)
rmse_ablation_2 = np.sqrt(mean_squared_error(y_val, y_pred_ablation_2))
print(f'Ablation 2 Performance (Increased Learning Rate): {rmse_ablation_2}')

# Determine which part of the code contributes the most to the overall performance
if rmse < rmse_ablation_1 and rmse < rmse_ablation_2:
    print('The original model configuration contributes the most to the overall performance.')
elif rmse_ablation_1 < rmse and rmse_ablation_1 < rmse_ablation_2:
    print('Reducing the number of estimators contributes the most to the overall performance.')
else:
    print('Increasing the learning rate contributes the most to the overall performance.')
