
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

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
    df['Spending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0).sum(axis=1)
    df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    return df

y = train['Transported'].astype(int)
X = engineer_features(train.drop(columns=['Transported']))
X_test = engineer_features(test.copy())

cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

model = Pipeline([
    ('prep', preprocess),
    ('clf', LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    ))
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
valid_acc = accuracy_score(y_valid, valid_pred)

test_pred = model.predict(X_test).astype(bool)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f"Final Validation Performance: {valid_acc:.6f}")
