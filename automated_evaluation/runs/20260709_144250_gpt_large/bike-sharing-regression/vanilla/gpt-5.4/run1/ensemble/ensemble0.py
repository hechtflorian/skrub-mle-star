
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

for df in [train, test]:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(int)

    # Safe engineered features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['quarter'] = df['datetime'].dt.quarter
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['temp_diff'] = df['temp'] - df['atemp']
    df['humidity_windspeed'] = df['humidity'] * df['windspeed']
    df['temp_humidity'] = df['temp'] * df['humidity']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

train = train.sort_values('datetime').reset_index(drop=True)
test = test.sort_values('datetime').reset_index(drop=True)

split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

baseline_features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear'
]

engineered_features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear',
    'is_weekend', 'quarter', 'dayofyear', 'temp_diff',
    'humidity_windspeed', 'temp_humidity',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
]

light_engineered_features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'hour',
    'dayofweek', 'weekofyear', 'is_weekend', 'quarter', 'dayofyear',
    'temp_diff', 'humidity_windspeed', 'temp_humidity',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
]

def rmsle_score(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def train_and_predict(features):
    X_train = train_part[features]
    y_train = np.log1p(train_part['count'])
    X_valid = valid_part[features]
    y_valid_raw = valid_part['count'].values

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    valid_pred_log = model.predict(X_valid)
    valid_pred = np.expm1(valid_pred_log).clip(0)

    X_full = train[features]
    y_full = np.log1p(train['count'])
    X_test = test[features]

    final_model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    final_model.fit(X_full, y_full)
    test_pred = np.expm1(final_model.predict(X_test)).clip(0)

    return valid_pred, test_pred, rmsle_score(y_valid_raw, valid_pred)

# Model A: original baseline
valid_pred_A, test_pred_A, rmsle_A = train_and_predict(baseline_features)

# Model B: stronger engineered feature variant
valid_pred_B, test_pred_B, rmsle_B = train_and_predict(engineered_features)

# Model C: lighter engineered variant
valid_pred_C, test_pred_C, rmsle_C = train_and_predict(light_engineered_features)

y_valid_raw = valid_part['count'].values

# Small manual grid for blend weights on natural count scale
blend_candidates = [
    (0.5, 0.5, 0.0),
    (0.4, 0.6, 0.0),
    (0.6, 0.4, 0.0),
    (0.4, 0.4, 0.2),
    (0.3, 0.5, 0.2),
    (0.5, 0.3, 0.2),
]

best_score = float('inf')
best_weights = None

for wA, wB, wC in blend_candidates:
    valid_blend = wA * valid_pred_A + wB * valid_pred_B + wC * valid_pred_C
    score = rmsle_score(y_valid_raw, valid_blend)
    if score < best_score:
        best_score = score
        best_weights = (wA, wB, wC)

wA, wB, wC = best_weights
final_test_pred = wA * test_pred_A + wB * test_pred_B + wC * test_pred_C

submission = pd.DataFrame({
    'datetime': test['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': final_test_pred
})
submission.to_csv('submission.csv', index=False)

print(f'Model A Validation RMSLE: {rmsle_A}')
print(f'Model B Validation RMSLE: {rmsle_B}')
print(f'Model C Validation RMSLE: {rmsle_C}')
print(f'Best Blend Weights: A={wA}, B={wB}, C={wC}')
print(f'Final Validation Performance: {best_score}')
