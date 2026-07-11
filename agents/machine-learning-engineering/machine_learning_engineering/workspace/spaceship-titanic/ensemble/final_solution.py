
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
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

def prepare_view_a(df):
    return engineer_features(df)

def prepare_view_b(df):
    df = engineer_features(df)
    for col in ['Deck', 'Side', 'Destination', 'HomePlanet', 'Group']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if 'Group' in df.columns:
        df = df.drop(columns=['Group'])
    return df

def build_model(X, seed):
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
            random_state=seed
        ))
    ])
    return model

def get_oof_predictions(X_train_full, y_train_full, seeds=(42, 52), n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train_full), dtype=float)

    for tr_idx, va_idx in skf.split(X_train_full, y_train_full):
        X_tr = X_train_full.iloc[tr_idx]
        y_tr = y_train_full.iloc[tr_idx]
        X_va = X_train_full.iloc[va_idx]

        fold_val_pred = np.zeros(len(va_idx), dtype=float)

        for seed in seeds:
            model = build_model(X_tr, seed)
            model.fit(X_tr, y_tr)
            fold_val_pred += model.predict_proba(X_va)[:, 1] / len(seeds)

        oof_pred[va_idx] = fold_val_pred

    return oof_pred

def fit_full_predict_test(X_tr, y_tr, X_te, seeds=(42, 52)):
    test_pred = np.zeros(len(X_te), dtype=float)
    for seed in seeds:
        model = build_model(X_tr, seed)
        model.fit(X_tr, y_tr)
        test_pred += model.predict_proba(X_te)[:, 1] / len(seeds)
    return test_pred

y = train['Transported'].astype(int)
X_raw = train.drop(columns=['Transported'])
X_test_raw = test.copy()

X_a = prepare_view_a(X_raw)
X_test_a = prepare_view_a(X_test_raw)

X_b = prepare_view_b(X_raw)
X_test_b = prepare_view_b(X_test_raw)

oof_a = get_oof_predictions(X_a, y, seeds=(42, 52), n_splits=5)
oof_b = get_oof_predictions(X_b, y, seeds=(42, 52), n_splits=5)

best_w = 0.5
best_oof_acc = -1.0
for w in np.arange(0.2, 0.81, 0.1):
    blend_oof = w * oof_a + (1.0 - w) * oof_b
    acc = accuracy_score(y, (blend_oof >= 0.5).astype(int))
    if acc > best_oof_acc:
        best_oof_acc = acc
        best_w = float(w)

best_threshold = 0.5
best_threshold_acc = -1.0
best_blend_oof = best_w * oof_a + (1.0 - best_w) * oof_b
for thr in np.arange(0.45, 0.551, 0.01):
    acc = accuracy_score(y, (best_blend_oof >= thr).astype(int))
    if acc > best_threshold_acc:
        best_threshold_acc = acc
        best_threshold = float(thr)

test_a_full = fit_full_predict_test(X_a, y, X_test_a, seeds=(42, 52))
test_b_full = fit_full_predict_test(X_b, y, X_test_b, seeds=(42, 52))

test_blend = best_w * test_a_full + (1.0 - best_w) * test_b_full
test_pred = (test_blend >= best_threshold).astype(bool)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_pred
})

os.makedirs('./final', exist_ok=True)
submission.to_csv('./final/submission.csv', index=False)

print(f"OOF Accuracy: {best_oof_acc:.6f}")
print(f"Blend Weight: {best_w:.2f}")
print(f"Threshold: {best_threshold:.2f}")
print("Saved submission to ./final/submission.csv")
