
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

# Load the datasets
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    # If running in an environment where the files are not in the root directory
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

# --- Preprocessing ---

# Separate target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Handle potential non-numeric entries and ensure 'total_bedrooms' is treated as numeric
for df in [X, test_df]:
    if 'total_bedrooms' in df.columns:
        df['total_bedrooms'] = pd.to_numeric(df['total_bedrooms'], errors='coerce')
        if 'total_bedrooms' not in numerical_features:
            numerical_features.append('total_bedrooms')

# Impute missing values for numerical features using the median
numerical_imputer = SimpleImputer(strategy='median')
X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])

# Create a copy of test_df for preprocessing
test_df_processed = test_df.copy()
if 'total_bedrooms' not in test_df_processed.columns:
     test_df_processed['total_bedrooms'] = np.nan # Ensure column exists

test_df_processed[numerical_features] = numerical_imputer.transform(test_df_processed[numerical_features]) # Use the imputer fitted on training data

# Drop rows where the target variable is missing (only for training data)
train_df.dropna(subset=['median_house_value'], inplace=True)

# --- Model Training ---

# Define features for both models
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

# Ensure all features used for modeling are present and imputed in X and test_df_processed
X = X[features]
test_df_processed = test_df_processed[features]


# Split data for validation for RandomForest
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf, y_train_rf)

# Train Linear Regression model (using the same split for consistent validation)
lr_model = LinearRegression()
lr_model.fit(X_train_rf, y_train_rf) # Train LR on the same training split as RF


# --- Model Evaluation ---

# Evaluate RandomForest model on the validation set
rf_val_predictions = rf_model.predict(X_val_rf)
rf_rmse_val = np.sqrt(mean_squared_error(y_val_rf, rf_val_predictions))

# Evaluate Linear Regression model on the validation set
lr_val_predictions = lr_model.predict(X_val_rf)
lr_rmse_val = np.sqrt(mean_squared_error(y_val_rf, lr_val_predictions))


# --- Ensembling ---

# Simple averaging ensemble
# Give more weight to the model that performed better on validation set (lower RMSE)
if rf_rmse_val < lr_rmse_val:
    # RandomForest is better, give it more weight
    ensemble_predictions_val = (0.6 * rf_val_predictions) + (0.4 * lr_val_predictions)
    ensemble_rmse_val = np.sqrt(mean_squared_error(y_val_rf, ensemble_predictions_val))
else:
    # Linear Regression is better or equal, give it more weight
    ensemble_predictions_val = (0.4 * rf_val_predictions) + (0.6 * lr_val_predictions)
    ensemble_rmse_val = np.sqrt(mean_squared_error(y_val_rf, ensemble_predictions_val))

final_validation_score = ensemble_rmse_val

print(f"Final Validation Performance: {final_validation_score}")

# --- Prediction on Test Data ---

# Predict on the processed test data using both models
rf_test_predictions = rf_model.predict(test_df_processed)
lr_test_predictions = lr_model.predict(test_df_processed)

# Create ensemble predictions for test data
# Use the same weights determined from validation performance
if rf_rmse_val < lr_rmse_val:
    ensemble_test_predictions = (0.6 * rf_test_predictions) + (0.4 * lr_test_predictions)
else:
    ensemble_test_predictions = (0.4 * rf_test_predictions) + (0.6 * lr_test_predictions)


# Ensure predictions are non-negative
ensemble_test_predictions[ensemble_test_predictions < 0] = 0

# Print the predictions in the required format (optional, as per original base solution, but not explicitly required by prompt for submission)
# for pred in ensemble_test_predictions:
#    print(f"{pred:.12f}")

# Create submission file (optional, but good practice for Kaggle)
submission_df = pd.DataFrame({'Id': test_df.index, 'median_house_value': ensemble_test_predictions})
submission_df.to_csv('submission.csv', index=False)
