
# Ensemble solution for median_house_value prediction
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]
X_test = test_df.copy()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Solution 1: Original Gradient Boosting model
def train_solution_1():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=42, max_depth=6, n_estimators=300))
    ])
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    test_predictions = pipeline.predict(X_test)
    return pipeline, val_rmse, test_predictions

# Solution 2: Modified Gradient Boosting with different hyperparameters
def train_solution_2():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=42, max_depth=4, n_estimators=500,
                                          learning_rate=0.05, min_samples_leaf=5))
    ])
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    test_predictions = pipeline.predict(X_test)
    return pipeline, val_rmse, test_predictions

# Train individual solutions
pipeline1, val_rmse_1, test_preds_1 = train_solution_1()
pipeline2, val_rmse_2, test_preds_2 = train_solution_2()

# Print individual validation performances
print(f"Solution 1 Validation RMSE: {val_rmse_1}")
print(f"Solution 2 Validation RMSE: {val_rmse_2}")

# Calculate weights based on inverse RMSE
weight_1 = 1 / val_rmse_1
weight_2 = 1 / val_rmse_2
total_weight = weight_1 + weight_2
final_weight_1 = weight_1 / total_weight
final_weight_2 = weight_2 / total_weight

# Create ensemble predictions for validation and test
ensemble_val_preds = (pipeline1.predict(X_val) * final_weight_1) + (pipeline2.predict(X_val) * final_weight_2)
ensemble_test_preds = (test_preds_1 * final_weight_1) + (test_preds_2 * final_weight_2)

# Calculate and print ensemble validation performance
final_validation_score = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
print(f"Final Validation Performance: {final_validation_score}")

# Save ensemble predictions
output = pd.DataFrame({"median_house_value": ensemble_test_preds})
output.to_csv("submission.csv", index=False, header=True)

# Save individual predictions for verification
pd.DataFrame({"median_house_value": test_preds_1}).to_csv("submission1.csv", index=False)
pd.DataFrame({"median_house_value": test_preds_2}).to_csv("submission2.csv", index=False)
