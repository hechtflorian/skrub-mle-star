
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


X = X.copy()
X_test = X_test.copy()

# Preserve native numeric missing handling in CatBoost; only normalize categorical missing values.
for c in cat_cols:
    X[c] = X[c].fillna('Missing').astype(str)
    X_test[c] = X_test[c].fillna('Missing').astype(str)

# Lightweight feature engineering: row-wise missing counts.
X['row_missing_count'] = X.isna().sum(axis=1)
X_test['row_missing_count'] = X_test.isna().sum(axis=1)

# Cast low-cardinality numeric/discrete columns to categorical so CatBoost can model them better.
low_card_num_as_cat = []
for c in num_cols:
    nunique = X[c].nunique(dropna=True)
    if nunique <= 20:
        low_card_num_as_cat.append(c)
        X[c] = X[c].astype('object').where(X[c].notna(), 'Missing').astype(str)
        X_test[c] = X_test[c].astype('object').where(X_test[c].notna(), 'Missing').astype(str)

# Missing-count feature can also be useful as categorical if it has low cardinality.
engineered_cat_cols = []
if X['row_missing_count'].nunique(dropna=True) <= 20:
    X['row_missing_count'] = X['row_missing_count'].astype(str)
    X_test['row_missing_count'] = X_test['row_missing_count'].astype(str)
    engineered_cat_cols.append('row_missing_count')

cat_features_final = list(dict.fromkeys(list(cat_cols) + low_card_num_as_cat + engineered_cat_cols))

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Logloss',
    verbose=0,
    random_state=RANDOM_STATE
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features_final,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

valid_proba = model.predict_proba(X_valid)[:, 1]

thresholds = np.linspace(0.05, 0.95, 181)
threshold_scores = [
    accuracy_score(y_valid, (valid_proba >= thr).astype(int))
    for thr in thresholds
]
best_threshold = thresholds[int(np.argmax(threshold_scores))]
final_validation_score = max(threshold_scores)

final_model = CatBoostClassifier(
    iterations=model.get_best_iteration() if model.get_best_iteration() is not None and model.get_best_iteration() > 0 else 500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Logloss',
    verbose=0,
    random_state=RANDOM_STATE
)

final_model.fit(X, y, cat_features=cat_features_final)

test_proba = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int).ravel()


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': np.where(test_pred == 1, 'Extrovert', 'Introvert')
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {final_validation_score:.6f}')
