
import os
import copy
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

train_path = os.path.join('.', 'input', 'train.csv')
train = pd.read_csv(train_path)

def engineer_features(df, use_engineering=True, use_spending=True):
    df = df.copy()

    if use_engineering:
        cabin_split = df['Cabin'].fillna('Unknown/0/U').str.split('/', expand=True)
        df['Deck'] = cabin_split[0]
        df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
        df['Side'] = cabin_split[2]
        df['Group'] = df['PassengerId'].str.split('_').str[0]

        if use_spending:
            df['Spending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0).sum(axis=1)

        df = df.drop(columns=['Cabin', 'Name', 'PassengerId'])
    else:
        df = df.drop(columns=['Name', 'PassengerId'])

    return df

def build_model(X, use_ohe=True, use_lgbm=True):
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
    if use_ohe:
        cat_steps.append(('oh', OneHotEncoder(handle_unknown='ignore')))

    preprocess = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(cat_steps), cat_cols)
    ])

    if use_lgbm:
        clf = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42
        )
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, random_state=42)

    return Pipeline([
        ('prep', preprocess),
        ('clf', clf)
    ])

y = train['Transported'].astype(int)

ablations = [
    {
        'name': 'baseline',
        'use_engineering': True,
        'use_spending': True,
        'use_ohe': True,
        'use_lgbm': True,
    },
    {
        'name': 'no_feature_engineering',
        'use_engineering': False,
        'use_spending': False,
        'use_ohe': True,
        'use_lgbm': True,
    },
    {
        'name': 'no_spending_feature',
        'use_engineering': True,
        'use_spending': False,
        'use_ohe': True,
        'use_lgbm': True,
    },
    {
        'name': 'no_one_hot_encoding',
        'use_engineering': True,
        'use_spending': True,
        'use_ohe': False,
        'use_lgbm': True,
    },
    {
        'name': 'replace_lgbm_with_logreg',
        'use_engineering': True,
        'use_spending': True,
        'use_ohe': True,
        'use_lgbm': False,
    },
]

results = {}

for ab in ablations:
    X = engineer_features(
        train.drop(columns=['Transported']),
        use_engineering=ab['use_engineering'],
        use_spending=ab['use_spending']
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        model = build_model(X, use_ohe=ab['use_ohe'], use_lgbm=ab['use_lgbm'])
        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)
        valid_acc = accuracy_score(y_valid, valid_pred)
        results[ab['name']] = valid_acc
        print(f"Ablation: {ab['name']}, Validation Accuracy: {valid_acc:.6f}")
    except Exception as e:
        results[ab['name']] = None
        print(f"Ablation: {ab['name']}, Validation Accuracy: FAILED, Error: {e}")

baseline_acc = results['baseline']
drops = []

for name, acc in results.items():
    if name == 'baseline' or acc is None:
        continue
    drop = baseline_acc - acc
    drops.append((name, drop))

if drops:
    most_important_part, largest_drop = max(drops, key=lambda x: x[1])
    print(f"Most important part: {most_important_part}, Accuracy drop vs baseline: {largest_drop:.6f}")
else:
    print("Most important part: could not be determined from the successful ablations.")
