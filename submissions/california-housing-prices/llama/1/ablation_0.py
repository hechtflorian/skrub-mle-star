
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

# Baseline model
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr_pred = gbr.predict(X_val)
rf_pred = rf.predict(X_val)
y_pred = (gbr_pred + rf_pred) / 2
rmse_baseline = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Baseline Performance: {rmse_baseline}')

# Ablation 1: Remove Random Forest Regressor
gbr_ablation1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_ablation1.fit(X_train, y_train)
gbr_pred_ablation1 = gbr_ablation1.predict(X_val)
rmse_ablation1 = np.sqrt(mean_squared_error(y_val, gbr_pred_ablation1))
print(f'Ablation 1 Performance (Remove Random Forest Regressor): {rmse_ablation1}')
print(f'Difference from Baseline: {rmse_ablation1 - rmse_baseline}')

# Ablation 2: Remove Gradient Boosting Regressor
rf_ablation2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_ablation2.fit(X_train, y_train)
rf_pred_ablation2 = rf_ablation2.predict(X_val)
rmse_ablation2 = np.sqrt(mean_squared_error(y_val, rf_pred_ablation2))
print(f'Ablation 2 Performance (Remove Gradient Boosting Regressor): {rmse_ablation2}')
print(f'Difference from Baseline: {rmse_ablation2 - rmse_baseline}')

if abs(rmse_ablation1 - rmse_baseline) > abs(rmse_ablation2 - rmse_baseline):
    print('The Random Forest Regressor contributes the most to the overall performance.')
else:
    print('The Gradient Boosting Regressor contributes the most to the overall performance.')
