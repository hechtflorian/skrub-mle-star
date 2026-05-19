
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        return False
    return True

try:
    import pytabkit.models
    from pytabkit.models import TabularModel
    use_pytabkit = True
except ImportError:
    print("pytabkit not available, falling back to XGBoost")
    if not install_package('xgboost'):
        raise ImportError("Neither pytabkit nor xgboost could be installed.")
    import xgboost as xgb
    use_pytabkit = False

train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')

X = train_data.drop(columns=['median_house_value'])
y = train_data['median_house_value']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
test_data_imputed = imputer.transform(test_data)

if use_pytabkit:
    model = TabularModel(task='regression', verbose=1)
    model.fit(X_train_imputed, y_train, X_val=X_val_imputed, y_val=y_val)
    val_preds = model.predict(X_val_imputed)
else:
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train)
    val_preds = model.predict(X_val_imputed)

val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f'Final Validation Performance: {val_rmse}')

test_preds = model.predict(test_data_imputed)
submission = pd.DataFrame({'median_house_value': test_preds})
submission.to_csv('submission.csv', index=False, header=True)
