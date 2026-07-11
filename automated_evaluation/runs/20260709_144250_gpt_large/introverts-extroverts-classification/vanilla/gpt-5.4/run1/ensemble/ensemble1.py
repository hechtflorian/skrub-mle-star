
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

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

# -----------------------------
# Model 1: preserve original solution
# -----------------------------
param_grid_1 = [
    {"depth": 6, "l2_leaf_reg": 3},
    {"depth": 8, "l2_leaf_reg": 5},
]

best_cv_score_1 = -np.inf
best_params_1 = None

for params in param_grid_1:
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
    if mean_score > best_cv_score_1:
        best_cv_score_1 = mean_score
        best_params_1 = params

oof_pred_1 = np.zeros(len(X), dtype=int)
oof_proba_1 = np.zeros(len(X), dtype=float)
test_proba_1 = np.zeros(len(X_test), dtype=float)

for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    final_model_1 = CatBoostClassifier(
        iterations=1200,
        depth=best_params_1["depth"],
        l2_leaf_reg=best_params_1["l2_leaf_reg"],
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='Accuracy',
        verbose=0,
        random_state=RANDOM_STATE + fold_idx
    )

    final_model_1.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=100
    )

    oof_pred_1[valid_idx] = final_model_1.predict(X_valid).astype(int).ravel()
    oof_proba_1[valid_idx] = final_model_1.predict_proba(X_valid)[:, 1]
    test_proba_1 += final_model_1.predict_proba(X_test)[:, 1] / cv.n_splits

# -----------------------------
# Model 2: same setup, slight variation to keep two-model ensemble
# -----------------------------
param_grid_2 = [
    {"depth": 5, "l2_leaf_reg": 3},
    {"depth": 7, "l2_leaf_reg": 7},
]

best_cv_score_2 = -np.inf
best_params_2 = None

for params in param_grid_2:
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(
            iterations=1500,
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            learning_rate=0.03,
            loss_function='Logloss',
            eval_metric='Accuracy',
            verbose=0,
            random_state=RANDOM_STATE + 100 + fold_idx
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
    if mean_score > best_cv_score_2:
        best_cv_score_2 = mean_score
        best_params_2 = params

oof_pred_2 = np.zeros(len(X), dtype=int)
oof_proba_2 = np.zeros(len(X), dtype=float)
test_proba_2 = np.zeros(len(X_test), dtype=float)

for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    final_model_2 = CatBoostClassifier(
        iterations=1500,
        depth=best_params_2["depth"],
        l2_leaf_reg=best_params_2["l2_leaf_reg"],
        learning_rate=0.03,
        loss_function='Logloss',
        eval_metric='Accuracy',
        verbose=0,
        random_state=RANDOM_STATE + 100 + fold_idx
    )

    final_model_2.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=100
    )

    oof_pred_2[valid_idx] = final_model_2.predict(X_valid).astype(int).ravel()
    oof_proba_2[valid_idx] = final_model_2.predict_proba(X_valid)[:, 1]
    test_proba_2 += final_model_2.predict_proba(X_test)[:, 1] / cv.n_splits

# -----------------------------
# Meta-level stacking on OOF only
# -----------------------------
meta_features_oof_full = pd.DataFrame({
    'p1': oof_proba_1,
    'p2': oof_proba_2,
    'p_mean': (oof_proba_1 + oof_proba_2) / 2.0,
    'p_diff': oof_proba_1 - oof_proba_2
})

meta_features_test_full = pd.DataFrame({
    'p1': test_proba_1,
    'p2': test_proba_2,
    'p_mean': (test_proba_1 + test_proba_2) / 2.0,
    'p_diff': test_proba_1 - test_proba_2
})

meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 999)

# Meta model 1: [p1, p2]
meta_oof_proba_1 = np.zeros(len(X), dtype=float)
meta_test_proba_1 = np.zeros(len(X_test), dtype=float)

for fold_idx, (train_idx, valid_idx) in enumerate(meta_cv.split(meta_features_oof_full, y)):
    X_meta_train = meta_features_oof_full[['p1', 'p2']].iloc[train_idx]
    X_meta_valid = meta_features_oof_full[['p1', 'p2']].iloc[valid_idx]
    y_meta_train = y.iloc[train_idx]

    meta_model_1 = LogisticRegression(random_state=RANDOM_STATE + fold_idx, max_iter=1000)
    meta_model_1.fit(X_meta_train, y_meta_train)

    meta_oof_proba_1[valid_idx] = meta_model_1.predict_proba(X_meta_valid)[:, 1]
    meta_test_proba_1 += meta_model_1.predict_proba(meta_features_test_full[['p1', 'p2']])[:, 1] / meta_cv.n_splits

# Meta model 2: [p1, p2, p_mean, p_diff]
meta_oof_proba_2 = np.zeros(len(X), dtype=float)
meta_test_proba_2 = np.zeros(len(X_test), dtype=float)

for fold_idx, (train_idx, valid_idx) in enumerate(meta_cv.split(meta_features_oof_full, y)):
    X_meta_train = meta_features_oof_full[['p1', 'p2', 'p_mean', 'p_diff']].iloc[train_idx]
    X_meta_valid = meta_features_oof_full[['p1', 'p2', 'p_mean', 'p_diff']].iloc[valid_idx]
    y_meta_train = y.iloc[train_idx]

    meta_model_2 = LogisticRegression(random_state=RANDOM_STATE + 1000 + fold_idx, max_iter=1000)
    meta_model_2.fit(X_meta_train, y_meta_train)

    meta_oof_proba_2[valid_idx] = meta_model_2.predict_proba(X_meta_valid)[:, 1]
    meta_test_proba_2 += meta_model_2.predict_proba(meta_features_test_full[['p1', 'p2', 'p_mean', 'p_diff']])[:, 1] / meta_cv.n_splits

meta_oof_proba = (meta_oof_proba_1 + meta_oof_proba_2) / 2.0
meta_test_proba = (meta_test_proba_1 + meta_test_proba_2) / 2.0

best_threshold = 0.5
best_score = -1.0

for threshold in np.arange(0.20, 0.8001, 0.005):
    preds = (meta_oof_proba >= threshold).astype(int)
    score = accuracy_score(y, preds)
    if score > best_score:
        best_score = score
        best_threshold = threshold

final_validation_score = best_score
test_pred = (meta_test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': np.where(test_pred == 1, 'Extrovert', 'Introvert')
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {final_validation_score}')
