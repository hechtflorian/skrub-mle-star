
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

from sklearn.model_selection import StratifiedKFold
import numpy as np

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

final_validation_score = best_cv_score

oof_pred = np.zeros(len(X), dtype=int)
test_pred_proba = np.zeros(len(X_test), dtype=float)

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

    oof_pred[valid_idx] = final_model.predict(X_valid).astype(int).ravel()
    test_pred_proba += final_model.predict_proba(X_test)[:, 1] / cv.n_splits

final_validation_score = accuracy_score(y, oof_pred)
test_pred = (test_pred_proba >= 0.5).astype(int)


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': np.where(test_pred == 1, 'Extrovert', 'Introvert')
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {final_validation_score:.6f}')
