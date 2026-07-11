
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

TARGET = 'NObeyesdad'
ID = 'id'

train_path = './input/train.csv'
test_path = './input/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

cat_cols = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'
]

X = train.drop(columns=[ID, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID])

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=200
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

valid_pred = model.predict(X_valid).ravel()
final_validation_score = accuracy_score(y_valid, valid_pred)
print(f'Final Validation Performance: {final_validation_score}')

final_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=200
)

final_model.fit(
    X,
    y,
    cat_features=cat_cols
)

test_pred = final_model.predict(X_test).ravel()
submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred
})
submission.to_csv('submission.csv', index=False)
