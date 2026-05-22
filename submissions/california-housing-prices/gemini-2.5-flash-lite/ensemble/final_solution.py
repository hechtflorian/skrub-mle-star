
import pandas as pd
from sklearn.model_selection import KFold
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

# --- Preprocessing Function to ensure consistency ---
def preprocess_data(X_data, y_data=None, imputer=None, is_test=False):
    X_processed = X_data.copy()

    # Ensure 'total_bedrooms' is numeric and handle potential errors
    if 'total_bedrooms' in X_processed.columns:
        X_processed['total_bedrooms'] = pd.to_numeric(X_processed['total_bedrooms'], errors='coerce')
        if 'total_bedrooms' not in numerical_features:
            numerical_features.append('total_bedrooms')

    # Imputation for numerical features
    if imputer is None:
        imputer = SimpleImputer(strategy='median')

    if not is_test:
        X_processed[numerical_features] = imputer.fit_transform(X_processed[numerical_features])
        if y_data is not None:
            return X_processed, y_data, imputer
        else:
            return X_processed, imputer
    else:
        # Impute test data using the imputer fitted on training data
        X_processed[numerical_features] = imputer.transform(X_processed[numerical_features])
        return X_processed

# --- Preprocess training data ---
X_processed_train, y_train, numerical_imputer = preprocess_data(X, y, is_test=False)

# --- Preprocess test data ---
# Make a copy of the original test_df to avoid modifying it directly in the loop
test_df_original_copy = test_df.copy()
X_processed_test = preprocess_data(test_df_original_copy, imputer=numerical_imputer, is_test=True)


# --- Ensemble Plan Implementation ---

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(X_processed_train.shape[0])
test_preds = np.zeros(X_processed_test.shape[0])

print(f"Starting training with {NFOLDS}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed_train, y_train)):
    print(f"--- Fold {fold+1}/{NFOLDS} ---")
    X_train, X_val = X_processed_train.iloc[train_idx], X_processed_train.iloc[val_idx]
    y_train_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train a RandomForestRegressor model with slightly varied parameters
    # Increased n_estimators and added max_depth as per the plan
    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42 + fold, n_jobs=-1)
    model.fit(X_train, y_train_fold)

    # Predict on validation set for OOF predictions
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds

    # Predict on the entire test set
    fold_test_preds = model.predict(X_processed_test)
    test_preds += fold_test_preds / NFOLDS # Average predictions across folds

    # Calculate and print validation performance for this fold
    fold_val_score = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold+1} Validation Performance (RMSE): {fold_val_score:.6f}")

# --- Final Evaluation ---
# Calculate the overall Out-of-Fold (OOF) performance
final_validation_score = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"Final Validation Performance: {final_validation_score:.6f}")


# --- Create Submission File ---
# Ensure predictions are non-negative
test_preds[test_preds < 0] = 0

# Create a submission DataFrame
submission_df = pd.DataFrame({'median_house_value': test_preds})

# Save the submission file
submission_df.to_csv('./final/submission.csv', index=False)

print("Submission file created successfully at ./final/submission.csv")
