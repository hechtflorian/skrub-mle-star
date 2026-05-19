
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from skrub import TableVectorizer
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, KFold

# Install required packages if not already installed
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package('skrub')
install_package('xgboost')
install_package('optuna')

# Load data
train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')

X = train_data.drop(columns=['median_house_value'])
y = train_data['median_house_value']

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Sample a subset for XGBoost hyperparameter tuning
np.random.seed(42)
sample_idx = np.random.choice(X_train.index, size=min(5000, len(X_train)), replace=False)
X_sample = X_train.loc[sample_idx]
y_sample = y_train.loc[sample_idx]

# Preprocess sample for XGBoost
preprocessor = TableVectorizer()
X_sample_vec = preprocessor.fit_transform(X_sample)
X_vec = preprocessor.transform(X_train)
X_val_vec = preprocessor.transform(X_val)
X_test_vec = preprocessor.transform(test_data)

def optimize_xgboost(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'tree_method': 'hist',
        'random_state': 42
    }

    model = xgb.XGBRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_sample_vec, y_sample, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(optimize_xgboost, n_trials=30, n_jobs=-1, callbacks=[optuna.study.MaxTrialsCallback(30, states=(optuna.trial.TrialState.COMPLETE,))])

best_params = study.best_params
best_params.update({
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42
})

# Feature importance-based pruning
secondary_model = xgb.XGBRegressor(**best_params)
secondary_model.fit(X_sample_vec, y_sample)
importance = secondary_model.feature_importances_
threshold = np.percentile(importance, 20)
important_features = importance > threshold
X_sample_pruned = X_sample_vec[:, important_features]
X_pruned = X_vec[:, important_features]
X_val_pruned = X_val_vec[:, important_features]
X_test_pruned = X_test_vec[:, important_features]

# Train secondary model on pruned features
secondary_model = xgb.XGBRegressor(**best_params)
secondary_model.fit(X_pruned, y_train)

# Build primary pipeline with TableVectorizer and RandomForest
primary_pipeline = Pipeline([
    ("preprocessor", TableVectorizer()),
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10))
])

# Train primary pipeline
primary_pipeline.fit(X_train, y_train)

# Predict on validation set for primary model
val_preds_rf = primary_pipeline.predict(X_val)
val_rmse_rf = np.sqrt(mean_squared_error(y_val, val_preds_rf))

# Predict on validation set for secondary model
val_preds_xgb = secondary_model.predict(X_val_pruned)
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, val_preds_xgb))

# Build ensemble model
ensemble_model = VotingRegressor([
    ('rf', primary_pipeline),
    ('xgb', Pipeline([
        ('preprocessor', TableVectorizer()),
        ('regressor', secondary_model)
    ]))
])

# Train ensemble model
ensemble_model.fit(X_train, y_train)

# Validate ensemble
val_preds_ensemble = ensemble_model.predict(X_val)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))

print(f'Random Forest Validation RMSE: {val_rmse_rf}')
print(f'XGBoost Validation RMSE: {val_rmse_xgb}')
print(f'Final Validation Performance: {final_validation_score}')

# Predict on test set
test_preds = ensemble_model.predict(test_data)

# Create submission
submission = pd.DataFrame({'median_house_value': test_preds})
submission.to_csv('submission.csv', index=False, header=True)
