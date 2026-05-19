
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

from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
import numpy as np

# Define numerical features (assuming it's already defined elsewhere)
# numerical_features = [...]

# Initialize a more sophisticated imputer for 'total_bedrooms' and potentially other numerical features
# We'll use KNNImputer for 'total_bedrooms' as it can capture relationships, and median for others as a fallback.
# If KNNImputer is too slow, SimpleImputer(strategy='mean') can be a good alternative.

# Impute general numerical features with median
numerical_imputer_median = SimpleImputer(strategy='median')
if numerical_features: # Ensure numerical_features is not empty
    X[numerical_features] = numerical_imputer_median.fit_transform(X[numerical_features])
    test_df_processed[numerical_features] = numerical_imputer_median.transform(test_df_processed[numerical_features])

# Special handling for 'total_bedrooms' using KNNImputer
if 'total_bedrooms' in numerical_features:
    # Initialize KNNImputer
    # n_neighbors can be tuned. A small number is usually sufficient.
    knn_imputer = KNNImputer(n_neighbors=5)

    # Fit KNNImputer on the training data's 'total_bedrooms' and transform both train and test
    # We need to ensure 'total_bedrooms' is treated as a feature for KNNImputer
    # It's best to fit KNNImputer on all numerical features where 'total_bedrooms' is present.
    # If 'total_bedrooms' is the only feature with NaNs that KNNImputer is meant for,
    # we can fit it on just that column, but fitting on all numerical features is more robust.

    # Identify numerical features excluding 'total_bedrooms' if we want to impute it separately
    other_numerical_features = [col for col in numerical_features if col != 'total_bedrooms']

    # Impute 'total_bedrooms' using KNNImputer
    if 'total_bedrooms' in X.columns:
        X['total_bedrooms'] = knn_imputer.fit_transform(X[[col for col in numerical_features if col in X.columns]])[:, numerical_features.index('total_bedrooms')]
    if 'total_bedrooms' in test_df_processed.columns:
        test_df_processed['total_bedrooms'] = knn_imputer.transform(test_df_processed[[col for col in numerical_features if col in test_df_processed.columns]])[:, numerical_features.index('total_bedrooms')]

elif 'total_bedrooms' in test_df_processed.columns and test_df_processed['total_bedrooms'].isnull().any():
    # If 'total_bedrooms' is not in numerical_features but has NaNs in test_df_processed,
    # this indicates a potential issue in feature definition or data loading.
    # We can attempt to impute it as a fallback.
    print("Warning: 'total_bedrooms' has NaNs in test_df_processed but was not listed in numerical_features. Attempting fallback imputation.")
    # Use median imputer as a fallback
    fallback_imputer = SimpleImputer(strategy='median')
    if 'total_bedrooms' in X.columns:
        fallback_imputer.fit(X[['total_bedrooms']])
        test_df_processed['total_bedrooms'] = fallback_imputer.transform(test_df_processed[['total_bedrooms']])
    else:
        print("Error: 'total_bedrooms' not found in training data for fallback imputation.")

# Ensure all columns in numerical_features are free of NaNs after imputation
# This is a final check and can catch any remaining issues.
for col in numerical_features:
    if X[col].isnull().any():
        print(f"Warning: NaNs found in training data column '{col}' after imputation. Using median.")
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
    if col in test_df_processed.columns and test_df_processed[col].isnull().any():
        print(f"Warning: NaNs found in test data column '{col}' after imputation. Using median from training data.")
        # Use the median from the training data to avoid data leakage
        if col in X.columns:
            median_val_train = X[col].median()
            test_df_processed[col].fillna(median_val_train, inplace=True)
        else: # If column is missing in training data, use test data median (less ideal)
            median_val_test = test_df_processed[col].median()
            test_df_processed[col].fillna(median_val_test, inplace=True)

# Handle categorical features (if any) - for simplicity, we'll assume no categorical features need complex handling for this specific error
# If there were categorical features, one-hot encoding or similar would be applied here.
# For this task, the error is related to numerical imputation, so we focus on that.



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
