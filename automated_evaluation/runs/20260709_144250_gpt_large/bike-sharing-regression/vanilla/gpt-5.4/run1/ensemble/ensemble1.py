
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
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

train = train.sort_values('datetime').reset_index(drop=True)
test = test.sort_values('datetime').reset_index(drop=True)

split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

y_train_log = np.log1p(train_part['count'])
y_valid_raw = valid_part['count'].values

features_A = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear'
]

features_B = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear', 'is_weekend'
]

features_C = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'humidity',
    'windspeed', 'month', 'hour', 'dayofweek', 'is_weekend'
]

model_params = dict(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

X_train_A = train_part[features_A]
X_valid_A = valid_part[features_A]
X_test_A = test[features_A]

X_train_B = train_part[features_B]
X_valid_B = valid_part[features_B]
X_test_B = test[features_B]

X_train_C = train_part[features_C]
X_valid_C = valid_part[features_C]
X_test_C = test[features_C]

model_A = LGBMRegressor(**model_params)
model_B = LGBMRegressor(**model_params)
model_C = LGBMRegressor(**model_params)

model_A.fit(X_train_A, y_train_log)
model_B.fit(X_train_B, y_train_log)
model_C.fit(X_train_C, y_train_log)

valid_pred_A = np.expm1(model_A.predict(X_valid_A)).clip(0)
test_pred_A = np.expm1(model_A.predict(X_test_A)).clip(0)

valid_pred_B = np.expm1(model_B.predict(X_valid_B)).clip(0)
test_pred_B = np.expm1(model_B.predict(X_test_B)).clip(0)

valid_pred_C = np.expm1(model_C.predict(X_valid_C)).clip(0)
test_pred_C = np.expm1(model_C.predict(X_test_C)).clip(0)

blend_candidates = [
    ('A0.2_B0.6_C0.2', (0.2, 0.6, 0.2)),
    ('A0.1_B0.7_C0.2', (0.1, 0.7, 0.2)),
    ('A0.0_B0.7_C0.3', (0.0, 0.7, 0.3)),
    ('A0.0_B0.8_C0.2', (0.0, 0.8, 0.2)),
]

best_base_name = None
best_base_score = float('inf')
best_base_valid = None
best_base_test = None

for name, (wa, wb, wc) in blend_candidates:
    base_valid = (wa * valid_pred_A + wb * valid_pred_B + wc * valid_pred_C).clip(0)
    base_test = (wa * test_pred_A + wb * test_pred_B + wc * test_pred_C).clip(0)
    score = np.sqrt(mean_squared_log_error(y_valid_raw, base_valid))
    if score < best_base_score:
        best_base_score = score
        best_base_name = name
        best_base_valid = base_valid
        best_base_test = base_test

corr_features = ['hour', 'dayofweek', 'month', 'weather', 'workingday', 'holiday', 'season', 'is_weekend']

residual_log = np.log1p(y_valid_raw) - np.log1p(best_base_valid.clip(0))

corr_model = LGBMRegressor(
    n_estimators=80,
    learning_rate=0.05,
    num_leaves=7,
    max_depth=3,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

corr_model.fit(valid_part[corr_features], residual_log)

corr_valid = corr_model.predict(valid_part[corr_features])
corr_test = corr_model.predict(test[corr_features])

best_alpha = None
best_final_score = float('inf')
best_final_valid = None
best_final_test = None

for alpha in [0.2, 0.35, 0.5]:
    final_valid = np.expm1(np.log1p(best_base_valid.clip(0)) + alpha * corr_valid).clip(0)
    final_test = np.expm1(np.log1p(best_base_test.clip(0)) + alpha * corr_test).clip(0)
    score = np.sqrt(mean_squared_log_error(y_valid_raw, final_valid))
    if score < best_final_score:
        best_final_score = score
        best_alpha = alpha
        best_final_valid = final_valid
        best_final_test = final_test

X_full_A = train[features_A]
X_full_B = train[features_B]
X_full_C = train[features_C]
y_full = np.log1p(train['count'])

final_model_A = LGBMRegressor(**model_params)
final_model_B = LGBMRegressor(**model_params)
final_model_C = LGBMRegressor(**model_params)

final_model_A.fit(X_full_A, y_full)
final_model_B.fit(X_full_B, y_full)
final_model_C.fit(X_full_C, y_full)

full_test_pred_A = np.expm1(final_model_A.predict(test[features_A])).clip(0)
full_test_pred_B = np.expm1(final_model_B.predict(test[features_B])).clip(0)
full_test_pred_C = np.expm1(final_model_C.predict(test[features_C])).clip(0)

name_to_weights = {name: weights for name, weights in blend_candidates}
wa, wb, wc = name_to_weights[best_base_name]
base_test_full = (wa * full_test_pred_A + wb * full_test_pred_B + wc * full_test_pred_C).clip(0)

final_test_pred = np.expm1(np.log1p(base_test_full.clip(0)) + best_alpha * corr_test).clip(0)

submission = pd.DataFrame({
    'datetime': test['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'count': final_test_pred
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {best_final_score}')
