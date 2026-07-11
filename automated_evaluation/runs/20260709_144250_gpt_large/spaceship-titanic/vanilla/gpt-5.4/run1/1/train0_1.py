
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols_lgb = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols_lgb = [c for c in X.columns if c not in cat_cols_lgb]

preprocess = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols_lgb),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols_lgb)
])

lgb_model = Pipeline([
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

lgb_model.fit(X_train, y_train)

X_cb = X.copy()
X_test_cb = X_test.copy()
X_train_cb = X_train.copy()
X_valid_cb = X_valid.copy()

for col in X_cb.columns:
    if X_cb[col].dtype == 'object':
        X_cb[col] = X_cb[col].fillna('Missing')
        X_test_cb[col] = X_test_cb[col].fillna('Missing')
        X_train_cb[col] = X_train_cb[col].fillna('Missing')
        X_valid_cb[col] = X_valid_cb[col].fillna('Missing')

numeric_cols_cb = [c for c in X_cb.columns if c not in X_cb.select_dtypes(include=['object']).columns]
for col in numeric_cols_cb:
    median_value = X_train_cb[col].median()
    X_cb[col] = X_cb[col].fillna(median_value)
    X_test_cb[col] = X_test_cb[col].fillna(median_value)
    X_train_cb[col] = X_train_cb[col].fillna(median_value)
    X_valid_cb[col] = X_valid_cb[col].fillna(median_value)

cat_cols_cb = X_cb.select_dtypes(include=['object']).columns.tolist()
cat_idx = [X_cb.columns.get_loc(c) for c in cat_cols_cb]

cb_model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=0,
    random_state=42
)

cb_model.fit(
    X_train_cb,
    y_train,
    cat_features=cat_idx,
    eval_set=(X_valid_cb, y_valid),
    use_best_model=True
)

lgb_valid_proba = lgb_model.predict_proba(X_valid)[:, 1]
cb_valid_proba = cb_model.predict_proba(X_valid_cb)[:, 1]
ensemble_valid_proba = 0.5 * lgb_valid_proba + 0.5 * cb_valid_proba
valid_pred = (ensemble_valid_proba >= 0.5).astype(int)
valid_acc = accuracy_score(y_valid, valid_pred)

lgb_test_proba = lgb_model.predict_proba(X_test)[:, 1]
cb_test_proba = cb_model.predict_proba(X_test_cb)[:, 1]
ensemble_test_proba = 0.5 * lgb_test_proba + 0.5 * cb_test_proba
test_pred = (ensemble_test_proba >= 0.5).astype(bool)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f"Final Validation Performance: {valid_acc:.6f}")
