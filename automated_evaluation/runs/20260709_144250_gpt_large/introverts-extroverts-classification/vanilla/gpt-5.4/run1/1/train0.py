
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

for c in cat_cols:
    X[c] = X[c].fillna('Missing').astype(str)
    X_test[c] = X_test[c].fillna('Missing').astype(str)

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

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=0,
    random_state=RANDOM_STATE
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

valid_pred = model.predict(X_valid).astype(int).ravel()
final_validation_score = accuracy_score(y_valid, valid_pred)

final_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=0,
    random_state=RANDOM_STATE
)

final_model.fit(X, y, cat_features=cat_cols)

test_pred = final_model.predict(X_test).astype(int).ravel()

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': np.where(test_pred == 1, 'Extrovert', 'Introvert')
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {final_validation_score:.6f}')
