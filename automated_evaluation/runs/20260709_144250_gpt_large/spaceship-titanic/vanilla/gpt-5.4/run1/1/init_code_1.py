
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = os.path.join('.', 'input', 'train.csv')
test_path = os.path.join('.', 'input', 'test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def engineer_features(df):
    df = df.copy()

    cabin_split = df['Cabin'].fillna('Unknown/0/U').str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
    df['Side'] = cabin_split[2]

    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

    name_split = df['Name'].fillna('Unknown Unknown').str.split(' ', n=1, expand=True)
    df['FirstName'] = name_split[0]
    df['LastName'] = name_split[1].fillna('Unknown')

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spend_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Spending'] = df[spend_cols].fillna(0).sum(axis=1)
    df['HasSpending'] = (df['Spending'] > 0).astype(str)
    df['IsAlone'] = (df['GroupSize'] == 1).astype(str)

    df['CryoSleep'] = df['CryoSleep'].astype('object')
    df['VIP'] = df['VIP'].astype('object')

    df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    return df

y = train['Transported'].astype(int)
X = engineer_features(train.drop(columns=['Transported']))
X_test = engineer_features(test.copy())

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('Missing')
        X_test[col] = X_test[col].fillna('Missing')

numeric_cols = [c for c in X.columns if c not in X.select_dtypes(include=['object']).columns]
for col in numeric_cols:
    median_value = X[col].median()
    X[col] = X[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=0,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_idx,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

valid_pred = model.predict(X_valid).astype(int).ravel()
valid_acc = accuracy_score(y_valid, valid_pred)

test_pred = model.predict(X_test).astype(bool).ravel()
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f"Final Validation Performance: {valid_acc:.6f}")
