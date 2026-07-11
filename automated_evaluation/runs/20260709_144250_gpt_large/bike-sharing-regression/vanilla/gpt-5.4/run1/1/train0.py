
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

features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear'
]

train = train.sort_values('datetime').reset_index(drop=True)

split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

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
rmsle = np.sqrt(mean_squared_log_error(y_valid_raw, valid_pred))

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

submission = pd.DataFrame({
    'datetime': test['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {rmsle}')
