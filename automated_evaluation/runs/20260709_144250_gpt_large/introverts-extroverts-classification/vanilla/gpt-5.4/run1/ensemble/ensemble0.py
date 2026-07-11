
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grid = [
    {"depth": 6, "l2_leaf_reg": 3},
    {"depth": 8, "l2_leaf_reg": 5},
]

best_cv_score = -np.inf
best_params = None

for params in param_grid:
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(
            iterations=1200,
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='Accuracy',
            verbose=0,
            random_state=RANDOM_STATE + fold_idx
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=100
        )

        valid_pred = model.predict(X_valid).astype(int).ravel()
        fold_scores.append(accuracy_score(y_valid, valid_pred))

    mean_score = np.mean(fold_scores)
    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_params = params

# Model A: original CV CatBoost with probability OOF
oof_proba_1 = np.zeros(len(X), dtype=float)
test_proba_1 = np.zeros(len(X_test), dtype=float)

for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    final_model = CatBoostClassifier(
        iterations=1200,
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='Accuracy',
        verbose=0,
        random_state=RANDOM_STATE + fold_idx
    )

    final_model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=100
    )

    oof_proba_1[valid_idx] = final_model.predict_proba(X_valid)[:, 1]
    test_proba_1 += final_model.predict_proba(X_test)[:, 1] / cv.n_splits

# Model B: feature-engineered CatBoost variant with minimal additional changes
X_fe = X.copy()
X_test_fe = X_test.copy()

if len(num_cols) >= 2:
    c1, c2 = num_cols[0], num_cols[1]
    X_fe[f'{c1}_plus_{c2}'] = X_fe[c1] + X_fe[c2]
    X_test_fe[f'{c1}_plus_{c2}'] = X_test_fe[c1] + X_test_fe[c2]
    X_fe[f'{c1}_minus_{c2}'] = X_fe[c1] - X_fe[c2]
    X_test_fe[f'{c1}_minus_{c2}'] = X_test_fe[c1] - X_test_fe[c2]

if len(num_cols) >= 1:
    c1 = num_cols[0]
    X_fe[f'{c1}_squared'] = X_fe[c1] ** 2
    X_test_fe[f'{c1}_squared'] = X_test_fe[c1] ** 2

cat_cols_fe = X_fe.select_dtypes(include=['object']).columns.tolist()

oof_proba_2 = np.zeros(len(X_fe), dtype=float)
test_proba_2 = np.zeros(len(X_test_fe), dtype=float)

for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_fe, y)):
    X_train, X_valid = X_fe.iloc[train_idx], X_fe.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model_b = CatBoostClassifier(
        iterations=1500,
        depth=7,
        l2_leaf_reg=5,
        learning_rate=0.04,
        loss_function='Logloss',
        eval_metric='Accuracy',
        verbose=0,
        random_state=RANDOM_STATE + 100 + fold_idx
    )

    model_b.fit(
        X_train,
        y_train,
        cat_features=cat_cols_fe,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=100
    )

    oof_proba_2[valid_idx] = model_b.predict_proba(X_valid)[:, 1]
    test_proba_2 += model_b.predict_proba(X_test_fe)[:, 1] / cv.n_splits

def find_best_threshold(y_true, proba):
    best_thr = 0.5
    best_score = -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        pred = (proba >= thr).astype(int)
        score = accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score

weight_candidates = [
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
]

best_weight_pair = None
best_threshold = None
best_ensemble_score = -1.0
best_ensemble_oof = None
best_ensemble_test = None

for w1, w2 in weight_candidates:
    ensemble_oof = w1 * oof_proba_1 + w2 * oof_proba_2
    ensemble_test = w1 * test_proba_1 + w2 * test_proba_2
    thr, score = find_best_threshold(y, ensemble_oof)

    if score > best_ensemble_score:
        best_ensemble_score = score
        best_weight_pair = (w1, w2)
        best_threshold = thr
        best_ensemble_oof = ensemble_oof
        best_ensemble_test = ensemble_test

final_validation_score = best_ensemble_score
test_pred = (best_ensemble_test >= best_threshold).astype(int)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': np.where(test_pred == 1, 'Extrovert', 'Introvert')
})
submission.to_csv('submission.csv', index=False)

print(f'Best Weights: {best_weight_pair}, Best Threshold: {best_threshold:.2f}')
print(f'Final Validation Performance: {final_validation_score}')
