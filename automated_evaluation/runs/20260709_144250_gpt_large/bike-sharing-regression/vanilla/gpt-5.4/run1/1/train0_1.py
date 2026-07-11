
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
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
    df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(int)

features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear'
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

lgbm_model = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=0,
    random_seed=42
)

lgbm_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train, cat_features=cat_features)

valid_pred_log_lgbm = lgbm_model.predict(X_valid)
valid_pred_log_cat = cat_model.predict(X_valid)
valid_pred_log = 0.5 * valid_pred_log_lgbm + 0.5 * valid_pred_log_cat
valid_pred = np.expm1(valid_pred_log).clip(0)
rmsle = np.sqrt(mean_squared_log_error(y_valid_raw, valid_pred))

X_full = train[features]
y_full = np.log1p(train['count'].values)
X_test = test[features]

final_lgbm_model = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=0,
    random_seed=42
)

final_lgbm_model.fit(X_full, y_full)
final_cat_model.fit(X_full, y_full, cat_features=cat_features)

test_pred_lgbm = np.expm1(final_lgbm_model.predict(X_test)).clip(0)
test_pred_cat = np.expm1(final_cat_model.predict(X_test)).clip(0)
test_pred = 0.5 * test_pred_lgbm + 0.5 * test_pred_cat

submission = pd.DataFrame({
    'datetime': test['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {rmsle}')
