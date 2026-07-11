
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

train = pd.read_csv('./input/train.csv')
train['datetime'] = pd.to_datetime(train['datetime'])

for df in [train]:
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(int)

base_features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'month', 'day', 'hour',
    'dayofweek', 'weekofyear'
]

train = train.sort_values('datetime').reset_index(drop=True)
split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

def evaluate_ablation(name, features, use_log_target=True):
    X_train = train_part[features]
    X_valid = valid_part[features]

    if use_log_target:
        y_train = np.log1p(train_part['count'])
    else:
        y_train = train_part['count']

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

    pred = model.predict(X_valid)
    if use_log_target:
        pred = np.expm1(pred)

    pred = np.clip(pred, 0, None)
    rmsle = np.sqrt(mean_squared_log_error(y_valid_raw, pred))
    print(f"{name}: RMSLE = {rmsle:.6f}")
    return rmsle

results = {}

results['baseline'] = evaluate_ablation(
    name='baseline',
    features=base_features,
    use_log_target=True
)

time_features_removed = [f for f in base_features if f not in ['hour', 'dayofweek', 'weekofyear', 'month']]
results['remove_key_time_features'] = evaluate_ablation(
    name='remove_key_time_features',
    features=time_features_removed,
    use_log_target=True
)

results['no_log_target'] = evaluate_ablation(
    name='no_log_target',
    features=base_features,
    use_log_target=False
)

baseline_score = results['baseline']
impacts = {
    k: v - baseline_score
    for k, v in results.items()
    if k != 'baseline'
}

worst_ablation = max(impacts, key=impacts.get)
print(f"Biggest contributor to performance: {worst_ablation} (RMSLE worsened by {impacts[worst_ablation]:.6f} vs baseline)")
