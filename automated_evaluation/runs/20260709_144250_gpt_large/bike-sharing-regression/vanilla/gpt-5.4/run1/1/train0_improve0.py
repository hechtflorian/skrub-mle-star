
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

    # stronger calendar signals
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df['is_month_start'] = df['datetime'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)
    df['is_year_start'] = df['datetime'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['datetime'].dt.is_year_end.astype(int)
    df['quarter'] = df['datetime'].dt.quarter

    # useful low-cost interactions
    df['hour_x_workingday'] = df['hour'] * df['workingday']
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']

    # cyclic encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# mark low-cardinality calendar fields as categorical for LightGBM
categorical_features = [
    'season', 'holiday', 'workingday', 'weather',
    'year', 'month', 'hour', 'dayofweek', 'weekofyear',
    'is_weekend', 'is_rush_hour', 'is_month_start', 'is_month_end',
    'is_year_start', 'is_year_end', 'quarter'
]

for df in [train, test]:
    for col in categorical_features:
        df[col] = df[col].astype('category')

features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear', 'quarter',
    'is_weekend', 'is_rush_hour', 'is_month_start', 'is_month_end',
    'is_year_start', 'is_year_end',
    'hour_x_workingday', 'hour_x_weekend',
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos'
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
