
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

# Impute general numerical features with median, excluding 'total_bedrooms' for now
numerical_features_excluding_bedrooms = [col for col in numerical_features if col != 'total_bedrooms']
if numerical_features_excluding_bedrooms:
    numerical_imputer_median = SimpleImputer(strategy='median')
    X[numerical_features_excluding_bedrooms] = numerical_imputer_median.fit_transform(X[numerical_features_excluding_bedrooms])
    test_df_processed[numerical_features_excluding_bedrooms] = numerical_imputer_median.transform(test_df_processed[numerical_features_excluding_bedrooms])

# Prepare data for predicting 'total_bedrooms'
# Use median-imputed numerical features (if any) and potentially other relevant features
features_for_bedrooms_prediction = numerical_features_excluding_bedrooms + ['some_other_relevant_feature'] # Example: add other features that might predict bedrooms
features_for_bedrooms_prediction = [f for f in features_for_bedrooms_prediction if f in X.columns and f in test_df_processed.columns] # Ensure features exist in both train and test

# Impute 'total_bedrooms' in training data first using median as a baseline if needed
if 'total_bedrooms' in X.columns and X['total_bedrooms'].isnull().any():
    # Use median to impute for training data if it's missing to create a base
    if 'total_bedrooms' not in numerical_features_excluding_bedrooms: # Ensure it's not already imputed if it was in the general list
        median_imputer_bedrooms = SimpleImputer(strategy='median')
        X['total_bedrooms'] = median_imputer_bedrooms.fit_transform(X[['total_bedrooms']])


# Train a model to predict 'total_bedrooms' using other features
# This is a simplified example using LinearRegression. A more complex model could be used.
# Ensure 'total_bedrooms' is not used as a feature to predict itself
if 'total_bedrooms' in features_for_bedrooms_prediction:
    features_for_bedrooms_prediction.remove('total_bedrooms')

# Filter features to ensure they are present in X
features_for_bedrooms_prediction = [f for f in features_for_bedrooms_prediction if f in X.columns]

# Create training data for the bedroom prediction model
X_train_bedrooms = X[features_for_bedrooms_prediction].copy()
y_train_bedrooms = X['total_bedrooms'].copy()

# Handle any remaining NaNs in features_for_bedrooms_prediction for training
# (This is a safeguard; ideally, they should be handled by general imputation)
for col in X_train_bedrooms.columns:
    if X_train_bedrooms[col].isnull().any():
        # Impute with median of that feature from training data
        median_val = X_train_bedrooms[col].median()
        X_train_bedrooms[col].fillna(median_val, inplace=True)

# Train the model
# Using LinearRegression as an example, a more complex model might be better.
from sklearn.linear_model import LinearRegression
bedroom_predictor_model = LinearRegression()
bedroom_predictor_model.fit(X_train_bedrooms, y_train_bedrooms)

# Predict missing 'total_bedrooms' in the test set
if 'total_bedrooms' in test_df_processed.columns and test_df_processed['total_bedrooms'].isnull().any():
    X_test_bedrooms = test_df_processed[features_for_bedrooms_prediction].copy()

    # Handle any NaNs in features_for_bedrooms_prediction for testing
    for col in X_test_bedrooms.columns:
        if X_test_bedrooms[col].isnull().any():
            # Impute with median of that feature from training data
            # Need to re-fit median imputer if it wasn't done for all numerical features
            # Or, more simply, use the median from the original X_train_bedrooms features
            if col in X_train_bedrooms.columns:
                median_val = X_train_bedrooms[col].median() # Use median from training data
                X_test_bedrooms[col].fillna(median_val, inplace=True)
            else: # Fallback if column somehow not in training features
                X_test_bedrooms[col].fillna(test_df_processed[col].median(), inplace=True)


    predicted_bedrooms = bedroom_predictor_model.predict(X_test_bedrooms)
    test_df_processed.loc[test_df_processed['total_bedrooms'].isnull(), 'total_bedrooms'] = predicted_bedrooms

# Ensure no NaNs remain in 'total_bedrooms' in test_df_processed after prediction
if test_df_processed['total_bedrooms'].isnull().any():
    # Fallback imputation if prediction failed for some reason
    fallback_median_bedrooms = X['total_bedrooms'].median() if 'total_bedrooms' in X.columns else 0
    test_df_processed['total_bedrooms'].fillna(fallback_median_bedrooms, inplace=True)


# Impute remaining numerical features in test_df_processed if any were missed or reintroduced as NaN
if numerical_features: # Ensure this list is up-to-date
    if 'total_bedrooms' in numerical_features and test_df_processed['total_bedrooms'].isnull().any():
         # This should not happen if the above logic worked, but as a safeguard
         if 'numerical_imputer_median' in locals():
             test_df_processed[numerical_features] = numerical_imputer_median.transform(test_df_processed[numerical_features])
         else: # Fallback if imputer was not created
             # Re-create and fit imputer
             temp_imputer = SimpleImputer(strategy='median')
             if 'total_bedrooms' in X.columns:
                 temp_imputer.fit(X[numerical_features])
                 test_df_processed[numerical_features] = temp_imputer.transform(test_df_processed[numerical_features])
             else:
                 print("Warning: 'total_bedrooms' not found in training data for fallback imputation.")




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
