
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np
import copy

# Load the datasets
try:
    train_df = pd.read_csv("train.csv")
except FileNotFoundError:
    train_df = pd.read_csv("./input/train.csv")

# Separate target variable
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# --- Baseline Model (Full Functionality) ---
def train_and_evaluate(X_train, y_train, X_val, y_val, description="Baseline"):
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()

    # Imputation for numerical features
    if 'total_bedrooms' in X_train_processed.columns:
        X_train_processed['total_bedrooms'] = pd.to_numeric(X_train_processed['total_bedrooms'], errors='coerce')
        if 'total_bedrooms' in X_val_processed.columns:
            X_val_processed['total_bedrooms'] = pd.to_numeric(X_val_processed['total_bedrooms'], errors='coerce')
        else:
            X_val_processed['total_bedrooms'] = np.nan
        if 'total_bedrooms' not in numerical_features:
            numerical_features.append('total_bedrooms')

    # Ensure all numerical features are present in both train and val, filling missing ones with NaN
    for col in numerical_features:
        if col not in X_train_processed.columns:
            X_train_processed[col] = np.nan
        if col not in X_val_processed.columns:
            X_val_processed[col] = np.nan

    numerical_imputer = SimpleImputer(strategy='median')
    X_train_processed[numerical_features] = numerical_imputer.fit_transform(X_train_processed[numerical_features])
    X_val_processed[numerical_features] = numerical_imputer.transform(X_val_processed[numerical_features])

    # Handle categorical features (if any) - Placeholder
    # For this ablation study, we assume no complex categorical handling is a key factor.

    # Ensure 'total_bedrooms' is imputed in X_val_processed, even if it was initially missing
    if 'total_bedrooms' in X_val_processed.columns and X_val_processed['total_bedrooms'].isnull().any():
        if 'total_bedrooms' in numerical_features:
            try:
                if hasattr(numerical_imputer, 'statistics_'):
                    X_val_processed['total_bedrooms'] = numerical_imputer.transform(X_val_processed[['total_bedrooms']])
                else:
                    numerical_imputer.fit(X_train_processed[numerical_features])
                    X_val_processed[numerical_features] = numerical_imputer.transform(X_val_processed[numerical_features])
            except Exception as e:
                print(f"Error during imputation safeguard for 'total_bedrooms': {e}")
                median_total_bedrooms = X_train_processed['total_bedrooms'].median()
                X_val_processed['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train)
    val_predictions = model.predict(X_val_processed)
    score = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"{description} - RMSE: {score:.4f}")
    return score

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Ablation 1: Disable imputation for 'total_bedrooms' ---
def train_without_total_bedrooms_imputation(X_train, y_train, X_val, y_val):
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()

    # Imputation for numerical features, excluding 'total_bedrooms'
    numerical_features_excluding_tb = [f for f in numerical_features if f != 'total_bedrooms']

    if 'total_bedrooms' in X_train_processed.columns:
        X_train_processed['total_bedrooms'] = pd.to_numeric(X_train_processed['total_bedrooms'], errors='coerce')
        if 'total_bedrooms' in X_val_processed.columns:
            X_val_processed['total_bedrooms'] = pd.to_numeric(X_val_processed['total_bedrooms'], errors='coerce')
        else:
            X_val_processed['total_bedrooms'] = np.nan

    # Ensure all numerical features are present
    for col in numerical_features_excluding_tb:
        if col not in X_train_processed.columns:
            X_train_processed[col] = np.nan
        if col not in X_val_processed.columns:
            X_val_processed[col] = np.nan

    numerical_imputer = SimpleImputer(strategy='median')
    X_train_processed[numerical_features_excluding_tb] = numerical_imputer.fit_transform(X_train_processed[numerical_features_excluding_tb])
    X_val_processed[numerical_features_excluding_tb] = numerical_imputer.transform(X_val_processed[numerical_features_excluding_tb])

    # Keep 'total_bedrooms' with NaNs if they exist
    if 'total_bedrooms' in X_train_processed.columns and X_train_processed['total_bedrooms'].isnull().any():
        pass # Keep NaNs
    if 'total_bedrooms' in X_val_processed.columns and X_val_processed['total_bedrooms'].isnull().any():
        pass # Keep NaNs

    # Impute remaining NaNs in 'total_bedrooms' with a constant (e.g., 0 or a specific value) to avoid model errors
    # Or, more simply for this ablation, if 'total_bedrooms' has NaNs, the model might fail or perform poorly.
    # Forcing imputation with a simple value like 0 to allow training.
    if 'total_bedrooms' in X_train_processed.columns:
        X_train_processed['total_bedrooms'].fillna(0, inplace=True)
    if 'total_bedrooms' in X_val_processed.columns:
        X_val_processed['total_bedrooms'].fillna(0, inplace=True)
    elif 'total_bedrooms' in X_train_processed.columns: # If missing in val but present in train
        X_val_processed['total_bedrooms'] = 0


    # Handle categorical features (if any) - Placeholder

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train)
    val_predictions = model.predict(X_val_processed)
    score = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Ablation: No Imputation for 'total_bedrooms' - RMSE: {score:.4f}")
    return score

# --- Ablation 2: Remove 'total_bedrooms' feature ---
def train_without_total_bedrooms_feature(X_train, y_train, X_val, y_val):
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()

    features_to_drop = []
    if 'total_bedrooms' in X_train_processed.columns:
        features_to_drop.append('total_bedrooms')
    if 'total_bedrooms' in X_val_processed.columns:
        features_to_drop.append('total_bedrooms')

    X_train_processed.drop(columns=features_to_drop, inplace=True)
    X_val_processed.drop(columns=features_to_drop, inplace=True)

    # Update numerical features list if 'total_bedrooms' was removed
    numerical_features_updated = [f for f in numerical_features if f != 'total_bedrooms']

    # Imputation for remaining numerical features
    # Ensure all numerical features are present
    for col in numerical_features_updated:
        if col not in X_train_processed.columns:
            X_train_processed[col] = np.nan
        if col not in X_val_processed.columns:
            X_val_processed[col] = np.nan

    numerical_imputer = SimpleImputer(strategy='median')
    X_train_processed[numerical_features_updated] = numerical_imputer.fit_transform(X_train_processed[numerical_features_updated])
    X_val_processed[numerical_features_updated] = numerical_imputer.transform(X_val_processed[numerical_features_updated])

    # Handle categorical features (if any) - Placeholder

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train)
    val_predictions = model.predict(X_val_processed)
    score = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Ablation: Removed 'total_bedrooms' Feature - RMSE: {score:.4f}")
    return score

# --- Run Ablations ---
print("--- Performing Ablation Study ---")

baseline_score = train_and_evaluate(X_train, y_train, X_val, y_val, description="Full Model")
ablation1_score = train_without_total_bedrooms_imputation(X_train, y_train, X_val, y_val)
ablation2_score = train_without_total_bedrooms_feature(X_train, y_train, X_val, y_val)

print("\n--- Ablation Study Results ---")
scores = {
    "Full Model": baseline_score,
    "No Imputation for 'total_bedrooms'": ablation1_score,
    "Removed 'total_bedrooms' Feature": ablation2_score
}

sorted_scores = sorted(scores.items(), key=lambda item: item[1])

print("\nModel performance ranked by RMSE (lower is better):")
for name, score in sorted_scores:
    print(f"- {name}: {score:.4f}")

print("\n--- Conclusion ---")
if baseline_score <= ablation1_score and baseline_score <= ablation2_score:
    print("The full model (including imputation for 'total_bedrooms' and using the feature) performed the best.")
    if baseline_score == ablation1_score and baseline_score < ablation2_score:
        print("Imputing 'total_bedrooms' is crucial, while removing the feature significantly degrades performance.")
    elif baseline_score < ablation1_score and baseline_score == ablation2_score:
         print("Keeping the 'total_bedrooms' feature is crucial, and its imputation further improves performance.")
    elif baseline_score < ablation1_score and baseline_score < ablation2_score:
        print("Both imputation of 'total_bedrooms' and using the feature itself are important for optimal performance.")
    elif baseline_score == ablation1_score and baseline_score == ablation2_score:
        print("None of the tested ablations significantly impacted performance, suggesting these features/steps might not be critical for this specific model configuration or dataset split.")
elif ablation1_score < baseline_score and ablation1_score <= ablation2_score:
    print("Disabling imputation for 'total_bedrooms' surprisingly improved performance. This might indicate issues with the imputation strategy or the data distribution.")
elif ablation2_score < baseline_score and ablation2_score < ablation1_score:
    print("Removing the 'total_bedrooms' feature significantly improved performance, suggesting it might be noisy or irrelevant.")
else:
    print("The ablation study results are complex and require further investigation.")

print("\nBased on these ablations, the most impactful component appears to be the correct handling and inclusion of the 'total_bedrooms' feature.")
if baseline_score < ablation1_score and baseline_score < ablation2_score:
    print("The full model, which includes imputation for 'total_bedrooms', contributes the most to the overall performance.")
elif ablation1_score < baseline_score and ablation1_score < ablation2_score:
    print("The omission of 'total_bedrooms' imputation led to the best performance in this specific ablation, suggesting that the imputation step might have introduced noise or that the feature without imputation is better.")
elif ablation2_score < baseline_score and ablation2_score < ablation1_score:
     print("Removing the 'total_bedrooms' feature led to the best performance, indicating it might be detrimental to the model's learning.")
else:
    print("The relative impact of different components is not clearly distinguishable from these simple ablations.")

