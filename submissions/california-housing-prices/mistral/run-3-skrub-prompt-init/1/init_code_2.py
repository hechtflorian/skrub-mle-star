
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Subsampling for faster training (if dataset is large)
sample_fraction = 0.1
if len(train_df) > 10000:
    train_df = train_df.sample(frac=sample_fraction, random_state=42)
if len(test_df) > 10000:
    test_df = test_df.sample(frac=sample_fraction, random_state=42)

# Feature engineering
def add_features(df):
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Prepare features and target
target = "median_house_value"
features = [col for col in train_df.columns if col != target]

X = train_df[features]
y = train_df[target]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
test_df_imputed = imputer.transform(test_df[features])

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Validate
val_predictions = model.predict(X_val)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Final Validation Performance: {final_validation_score}")

# Predict on test set
test_predictions = model.predict(test_df_imputed)

# Prepare submission
submission = pd.DataFrame({"median_house_value": test_predictions})
submission.to_csv("submission.csv", index=False, header=True)
