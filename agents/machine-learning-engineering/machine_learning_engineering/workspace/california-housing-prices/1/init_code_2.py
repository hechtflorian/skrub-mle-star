
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skrub import TableVectorizer

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_df.drop(columns=["median_house_value"])
y = train_df["median_house_value"]

# Subsampling for faster iteration (optional)
X = X.sample(frac=0.1, random_state=42)
y = y.loc[X.index]

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline with TableVectorizer and RandomForest
pipeline = Pipeline([
    ("preprocessor", TableVectorizer()),
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10))
])

# Train
pipeline.fit(X_train, y_train)

# Validate
val_preds = pipeline.predict(X_val)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Final Validation Performance: {final_validation_score}")

# Predict on test set
test_preds = pipeline.predict(test_df)
submission = pd.DataFrame({"median_house_value": test_preds})
submission.to_csv("submission.csv", index=False, header=True)
