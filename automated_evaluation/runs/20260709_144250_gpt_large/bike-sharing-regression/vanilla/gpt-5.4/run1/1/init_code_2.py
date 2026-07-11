
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
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

features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour', 'dayofweek'
]
cat_features = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'day', 'hour', 'dayofweek']

train = train.sort_values('datetime').reset_index(drop=True)
test = test.sort_values('datetime').reset_index(drop=True)

split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

X_train = train_part[features]
y_train = np.log1p(train_part['count'].values)

X_valid = valid_part[features]
y_valid_raw = valid_part['count'].values

model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=0,
    random_seed=42
)

model.fit(X_train, y_train, cat_features=cat_features)

valid_pred_log = model.predict(X_valid)
valid_pred = np.expm1(valid_pred_log).clip(0)
rmsle = np.sqrt(mean_squared_log_error(y_valid_raw, valid_pred))

X_full = train[features]
y_full = np.log1p(train['count'].values)
X_test = test[features]

final_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=0,
    random_seed=42
)

final_model.fit(X_full, y_full, cat_features=cat_features)

test_pred = np.expm1(final_model.predict(X_test)).clip(0)

submission = pd.DataFrame({
    'datetime': test['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {rmsle}')
