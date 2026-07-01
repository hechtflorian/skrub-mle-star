
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Separate target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Imputation for numerical features
# Create a copy of the test_df to avoid modifying the original test set directly
test_df_processed = test_df.copy()

# Ensure 'total_bedrooms' is in numerical_features if it exists and is numeric,
# and handle potential non-numeric entries before imputation.
if 'total_bedrooms' in X.columns:
    X['total_bedrooms'] = pd.to_numeric(X['total_bedrooms'], errors='coerce')
    if 'total_bedrooms' in test_df_processed.columns:
        test_df_processed['total_bedrooms'] = pd.to_numeric(test_df_processed['total_bedrooms'], errors='coerce')
    else:
        # If 'total_bedrooms' is missing in test_df_processed, add it with NaN values
        test_df_processed['total_bedrooms'] = np.nan
    if 'total_bedrooms' not in numerical_features:
        numerical_features.append('total_bedrooms')


# Impute missing values for numerical features using the median
numerical_imputer = SimpleImputer(strategy='median')
X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])
test_df_processed[numerical_features] = numerical_imputer.transform(test_df_processed[numerical_features]) # Use the imputer fitted on training data


# Handle categorical features (if any) - for simplicity, we'll assume no categorical features need complex handling for this specific error
# If there were categorical features, one-hot encoding or similar would be applied here.
# For this task, the error is related to numerical imputation, so we focus on that.


# Check if 'total_bedrooms' exists in the processed test_df and impute if necessary
# This is a safeguard and should ideally be covered by the general numerical imputation above
if 'total_bedrooms' in test_df_processed.columns and test_df_processed['total_bedrooms'].isnull().any():
    # Re-initialize imputer if needed or use the existing one if it was fit
    if 'numerical_imputer' not in locals():
        numerical_imputer = SimpleImputer(strategy='median')
        # Fit on training data's total_bedrooms, then transform test data
        # This is a more robust way if the above numerical imputation missed it
        if 'total_bedrooms' in X.columns:
             numerical_imputer.fit(X[['total_bedrooms']])
             test_df_processed['total_bedrooms'] = numerical_imputer.transform(test_df_processed[['total_bedrooms']])
        else: # Fallback if X also somehow lost total_bedrooms
             print("Warning: 'total_bedrooms' not found in training data for imputation.")


# For demonstration purposes, let's create a dummy test_df_processed if it was empty or had issues
# This part is mainly to ensure the code runs without errors related to missing columns in test_df_processed
if test_df_processed.empty:
    print("Warning: test_df_processed is empty. Creating dummy data for 'total_bedrooms'.")
    test_df_processed = pd.DataFrame(columns=X.columns) # Ensure all columns from X are present
    test_df_processed['total_bedrooms'] = np.nan # Add missing column

# Ensure 'total_bedrooms' is present in test_df_processed for imputation
if 'total_bedrooms' not in test_df_processed.columns:
    test_df_processed['total_bedrooms'] = np.nan # Add it if it was completely missing

# Re-apply imputation to ensure 'total_bedrooms' is handled in test_df_processed
# This step reiterates the imputation on numerical features for test_df_processed,
# ensuring that if 'total_bedrooms' was initially missing or problematic, it gets imputed.
if 'total_bedrooms' in numerical_features:
    if test_df_processed['total_bedrooms'].isnull().any():
        # Ensure the imputer is fitted on training data and used for test data
        # This block is a safeguard to ensure the imputation logic is applied correctly
        # even if the initial imputation step had issues.
        try:
            # Check if imputer was already fitted
            if hasattr(numerical_imputer, 'statistics_'):
                test_df_processed['total_bedrooms'] = numerical_imputer.transform(test_df_processed[['total_bedrooms']])
            else: # If not fitted, fit on training data and transform test data
                numerical_imputer.fit(X[numerical_features])
                test_df_processed[numerical_features] = numerical_imputer.transform(test_df_processed[numerical_features])
        except Exception as e:
            print(f"Error during final imputation safeguard for 'total_bedrooms': {e}")
            # Fallback: Impute with median of training data if imputer fails
            median_total_bedrooms = X['total_bedrooms'].median()
            test_df_processed['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
else:
    print("Warning: 'total_bedrooms' is not in numerical_features, skipping its imputation in test_df_processed.")


# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Predict on the processed test data
predictions = model.predict(test_df_processed)

# Ensure predictions are non-negative
predictions[predictions < 0] = 0

# Print the predictions in the required format
for pred in predictions:
    print(f"{pred:.12f}")

# Example of calculating final validation performance (using a small portion of training data as validation)
# This is for demonstration and might not reflect the actual test performance.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model_val = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_val.fit(X_train, y_train)
val_predictions = model_val.predict(X_val)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Final Validation Performance: {final_validation_score}")
