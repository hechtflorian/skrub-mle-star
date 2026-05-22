
# Single-file self-contained Python script for median_house_value prediction
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Subsampling for faster iteration (if dataset is large)
sample_size = min(10000, len(train_df))
train_df = train_df.sample(sample_size, random_state=42) if len(train_df) > 1000 else train_df

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]
X_test = test_df.copy()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline to handle missing values, scaling, feature selection, and modeling
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("selector", SelectFromModel(estimator=RandomForestRegressor(random_state=42, max_depth=6, n_estimators=50, n_jobs=-1), threshold="median")),
    ("model", GradientBoostingRegressor(random_state=42, max_depth=5, n_estimators=200))
])

# Train the model
pipeline.fit(X_train, y_train)

# Validate the model
val_predictions = pipeline.predict(X_val)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Final Validation Performance: {final_validation_score}")

# Predict on test set
test_predictions = pipeline.predict(X_test)

# Save predictions in the required format
output = pd.DataFrame({"median_house_value": test_predictions})
output.to_csv("submission.csv", index=False, header=True)
