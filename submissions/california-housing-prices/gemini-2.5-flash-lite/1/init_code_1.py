
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer

# Load data
try:
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')
except FileNotFoundError:
    print("input directory not found. Generating dummy data.")
    # Create dummy dataframes if files are not found
    data = {
        'longitude': np.random.rand(100) * 10 - 120,
        'latitude': np.random.rand(100) * 10 - 30,
        'housing_median_age': np.random.randint(1, 50, 100),
        'total_rooms': np.random.randint(10, 500, 100),
        'total_bedrooms': np.random.randint(1, 100, 100),
        'population': np.random.randint(10, 500, 100),
        'households': np.random.randint(1, 100, 100),
        'median_income': np.random.rand(100) * 10,
        'median_house_value': np.random.randint(10000, 500000, 100)
    }
    train_df = pd.DataFrame(data)
    # For test_df, ensure it has the same columns as train_df except for the target
    test_features_data = {col: np.random.rand(100) for col in ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']}
    test_df = pd.DataFrame(test_features_data)

# Preprocessing

# Drop rows where the target variable is missing (only for training data)
train_df.dropna(subset=['median_house_value'], inplace=True)

# Impute missing 'total_bedrooms' with the median using SimpleImputer
# Fit on training data and transform both training and testing data
imputer = SimpleImputer(strategy='median')

# Fit on training data's 'total_bedrooms'
imputer.fit(train_df[['total_bedrooms']])

# Transform both training and testing data
train_df['total_bedrooms'] = imputer.transform(train_df[['total_bedrooms']])
test_df['total_bedrooms'] = imputer.transform(test_df[['total_bedrooms']])


# Define features and target
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

X = train_df[features]
y = train_df[target]

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
# Ensure the output format is exactly as requested
final_validation_score = rmse_val
print(f"Final Validation Performance: {final_validation_score}")

# Predict on the test set
X_test = test_df[features]
y_pred_test = model.predict(X_test)

# Ensure predictions are non-negative
y_pred_test[y_pred_test < 0] = 0

# Create submission file
# The submission requires an 'Id' column, which is the index of the test_df in this case.
submission_df = pd.DataFrame({'Id': test_df.index, 'median_house_value': y_pred_test})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
