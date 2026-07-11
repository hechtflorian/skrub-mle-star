
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

    # Demand-shape interaction features
    df['hour_dayofweek'] = df['hour'].astype(str) + '_' + df['dayofweek'].astype(str)
    df['hour_workingday'] = df['hour'].astype(str) + '_' + df['workingday'].astype(str)
    df['season_hour'] = df['season'].astype(str) + '_' + df['hour'].astype(str)
    df['weather_hour'] = df['weather'].astype(str) + '_' + df['hour'].astype(str)

    # Coarser intraday regime
    df['part_of_day'] = pd.cut(
        df['hour'],
        bins=[-1, 5, 10, 15, 19, 23],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # Cheap nonlinear weather-behavior transforms
    df['feels_like_diff'] = df['temp'] - df['atemp']
    df['humidity_sq'] = df['humidity'] ** 2
    df['windspeed_sq'] = df['windspeed'] ** 2
    df['temp_x_humidity'] = df['temp'] * df['humidity']

# Frequency encodings for compact interaction groups
freq_cols = ['hour_dayofweek', 'hour_workingday', 'season_hour', 'weather_hour', 'part_of_day']
for col in freq_cols:
    freq_map = train[col].value_counts(normalize=True)
    train[f'{col}_freq'] = train[col].map(freq_map).astype(float)
    test[f'{col}_freq'] = test[col].map(freq_map).fillna(0).astype(float)

# Out-of-fold target encodings for low-cardinality interaction groups
target_col = None
for candidate in ['count', 'registered', 'casual']:
    if candidate in train.columns:
        target_col = candidate
        break

te_cols = []
if target_col is not None:
    from sklearn.model_selection import KFold

    te_cols = ['hour_dayofweek', 'hour_workingday', 'season_hour', 'weather_hour']
    global_mean = train[target_col].mean()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in te_cols:
        train[f'{col}_te'] = np.nan

        for tr_idx, val_idx in kf.split(train):
            tr_part = train.iloc[tr_idx]
            val_part = train.iloc[val_idx]

            means = tr_part.groupby(col)[target_col].mean()
            train.iloc[val_idx, train.columns.get_loc(f'{col}_te')] = val_part[col].map(means)

        train[f'{col}_te'] = train[f'{col}_te'].fillna(global_mean).astype(float)

        full_means = train.groupby(col)[target_col].mean()
        test[f'{col}_te'] = test[col].map(full_means).fillna(global_mean).astype(float)

features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear',
    'part_of_day', 'feels_like_diff', 'humidity_sq', 'windspeed_sq',
    'temp_x_humidity',
    'hour_dayofweek_freq', 'hour_workingday_freq', 'season_hour_freq',
    'weather_hour_freq', 'part_of_day_freq'
] + [f'{col}_te' for col in te_cols]


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
