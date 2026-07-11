
import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
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

def get_oof_and_test_predictions(X_train_full, y_train_full, X_test_full, seeds=(42, 52), n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train_full), dtype=float)
    test_pred = np.zeros(len(X_test_full), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_tr = X_train_full.iloc[tr_idx]
        y_tr = y_train_full.iloc[tr_idx]
        X_va = X_train_full.iloc[va_idx]

        fold_val_pred = np.zeros(len(va_idx), dtype=float)
        fold_test_pred = np.zeros(len(X_test_full), dtype=float)

        for seed in seeds:
            model = build_model(X_tr, seed)
            model.fit(X_tr, y_tr)
            fold_val_pred += model.predict_proba(X_va)[:, 1] / len(seeds)
            fold_test_pred += model.predict_proba(X_test_full)[:, 1] / len(seeds)

        oof_pred[va_idx] = fold_val_pred
        test_pred += fold_test_pred / n_splits

    return oof_pred, test_pred

# Hold-out split for reported validation metric
y = train['Transported'].astype(int)
X_raw = train.drop(columns=['Transported'])
X_test_raw = test.copy()

X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# Prepare both views
X_train_a = prepare_view_a(X_train_raw)
X_valid_a = prepare_view_a(X_valid_raw)
X_test_a = prepare_view_a(X_test_raw)

X_train_b = prepare_view_b(X_train_raw)
X_valid_b = prepare_view_b(X_valid_raw)
X_test_b = prepare_view_b(X_test_raw)

# OOF on training split only
oof_a, test_a = get_oof_and_test_predictions(X_train_a, y_train, X_test_a, seeds=(42, 52), n_splits=5)
oof_b, test_b = get_oof_and_test_predictions(X_train_b, y_train, X_test_b, seeds=(42, 52), n_splits=5)

acc_a = accuracy_score(y_train, (oof_a >= 0.5).astype(int))
acc_b = accuracy_score(y_train, (oof_b >= 0.5).astype(int))

best_w = 0.5
best_oof_acc = -1.0
for w in np.arange(0.2, 0.81, 0.1):
    blend_oof = w * oof_a + (1.0 - w) * oof_b
    acc = accuracy_score(y_train, (blend_oof >= 0.5).astype(int))
    if acc > best_oof_acc:
        best_oof_acc = acc
        best_w = float(w)

best_threshold = 0.5
best_threshold_acc = -1.0
best_blend_oof = best_w * oof_a + (1.0 - best_w) * oof_b
for thr in np.arange(0.45, 0.551, 0.01):
    acc = accuracy_score(y_train, (best_blend_oof >= thr).astype(int))
    if acc > best_threshold_acc:
        best_threshold_acc = acc
        best_threshold = float(thr)

# Train full models on training split and predict validation/test
def fit_full_predict(X_tr, y_tr, X_va, X_te, seeds=(42, 52)):
    val_pred = np.zeros(len(X_va), dtype=float)
    test_pred = np.zeros(len(X_te), dtype=float)
    for seed in seeds:
        model = build_model(X_tr, seed)
        model.fit(X_tr, y_tr)
        val_pred += model.predict_proba(X_va)[:, 1] / len(seeds)
        test_pred += model.predict_proba(X_te)[:, 1] / len(seeds)
    return val_pred, test_pred

valid_a, test_a_full = fit_full_predict(X_train_a, y_train, X_valid_a, X_test_a, seeds=(42, 52))
valid_b, test_b_full = fit_full_predict(X_train_b, y_train, X_valid_b, X_test_b, seeds=(42, 52))

valid_blend = best_w * valid_a + (1.0 - best_w) * valid_b
valid_pred = (valid_blend >= best_threshold).astype(int)
valid_acc = accuracy_score(y_valid, valid_pred)

test_blend = best_w * test_a_full + (1.0 - best_w) * test_b_full
test_pred = (test_blend >= best_threshold).astype(bool)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f"Final Validation Performance: {valid_acc:.6f}")
