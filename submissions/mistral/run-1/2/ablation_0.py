
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import catboost as cb

# Load data
train = pd.read_csv('./input/train.csv')

# Base model function
def train_base_model(X_train, X_val, y_train, y_val, feature_engineering=True):
    if feature_engineering:
        # Feature engineering
        X_train['rooms_per_household'] = X_train['total_rooms'] / X_train['households']
        X_train['bedrooms_per_room'] = X_train['total_bedrooms'] / X_train['total_rooms']
        X_train['population_per_household'] = X_train['population'] / X_train['households']

        X_val['rooms_per_household'] = X_val['total_rooms'] / X_val['households']
        X_val['bedrooms_per_room'] = X_val['total_bedrooms'] / X_val['total_rooms']
        X_val['population_per_household'] = X_val['population'] / X_val['households']

    # Train CatBoost model
    model = cb.CatBoostRegressor(random_state=42, verbose=0)
    model.fit(X_train, y_train)

    # Predict and evaluate
    val_preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_preds) ** 0.5
    return rmse

# Features and target
X = train.drop('median_house_value', axis=1)
y = train['median_house_value']

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ablation 1: No feature engineering
print("Running Ablation 1: No feature engineering")
X_train_orig = X_train.copy()
X_val_orig = X_val.copy()
rmse_no_fe = train_base_model(X_train_orig, X_val_orig, y_train, y_val, feature_engineering=False)
print(f"Ablation 1 RMSE (No feature engineering): {rmse_no_fe}")

# Ablation 2: Base model with feature engineering
print("\nRunning Ablation 2: Base model with feature engineering")
rmse_base = train_base_model(X_train, X_val, y_train, y_val, feature_engineering=True)
print(f"Ablation 2 RMSE (Base model with feature engineering): {rmse_base}")

# Compare performance
print("\nPerformance Comparison:")
print(f"Feature engineering impact: {rmse_no_fe - rmse_base:.4f} RMSE improvement")
if rmse_base < rmse_no_fe:
    print("Feature engineering contributes significantly to model performance.")
else:
    print("Feature engineering does not improve model performance.")
