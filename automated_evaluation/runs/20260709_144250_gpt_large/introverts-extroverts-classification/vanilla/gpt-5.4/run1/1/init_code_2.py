
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

train_path = os.path.join('.', 'input', 'train.csv')
test_path = os.path.join('.', 'input', 'test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = (train['Personality'] == 'Extrovert').astype(int)
X = train.drop(columns=['Personality']).copy()
X_test = test.copy()

cat_cols = X.select_dtypes(include='object').columns.tolist()

for df in [X, X_test]:
    for c in cat_cols:
        df[c] = df[c].fillna('Missing').astype('category')

num_cols = X.select_dtypes(exclude='category').columns.tolist()
num_cols = [c for c in num_cols if c != 'id']

for c in num_cols:
    median_value = X[c].median()
    X[c] = X[c].fillna(median_value)
    X_test[c] = X_test[c].fillna(median_value)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
final_validation_score = accuracy_score(y_valid, valid_pred)

final_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE
)

final_model.fit(X, y)
test_pred = final_model.predict(X_test)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': np.where(test_pred == 1, 'Extrovert', 'Introvert')
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {final_validation_score:.6f}')
