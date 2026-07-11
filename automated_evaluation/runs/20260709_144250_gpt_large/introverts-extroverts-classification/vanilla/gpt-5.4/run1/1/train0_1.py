
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
num_cols = [c for c in X.columns if c not in cat_cols and c != 'id']

X_cb = X.copy()
X_test_cb = X_test.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].fillna('Missing').astype(str)
    X_test_cb[c] = X_test_cb[c].fillna('Missing').astype(str)

for c in num_cols:
    median_value = X_cb[c].median()
    X_cb[c] = X_cb[c].fillna(median_value)
    X_test_cb[c] = X_test_cb[c].fillna(median_value)

X_lgb = X.copy()
X_test_lgb = X_test.copy()
for df in [X_lgb, X_test_lgb]:
    for c in cat_cols:
        df[c] = df[c].fillna('Missing').astype('category')

for c in num_cols:
    median_value = X_lgb[c].median()
    X_lgb[c] = X_lgb[c].fillna(median_value)
    X_test_lgb[c] = X_test_lgb[c].fillna(median_value)

train_idx, valid_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train_cb = X_cb.iloc[train_idx]
X_valid_cb = X_cb.iloc[valid_idx]
X_train_lgb = X_lgb.iloc[train_idx]
X_valid_lgb = X_lgb.iloc[valid_idx]
y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=0,
    random_state=RANDOM_STATE
)

lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE
)

cat_model.fit(
    X_train_cb,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid_cb, y_valid),
    use_best_model=True
)

lgb_model.fit(X_train_lgb, y_train)

cat_valid_proba = cat_model.predict_proba(X_valid_cb)[:, 1]
lgb_valid_proba = lgb_model.predict_proba(X_valid_lgb)[:, 1]
ensemble_valid_proba = 0.5 * cat_valid_proba + 0.5 * lgb_valid_proba
valid_pred = (ensemble_valid_proba >= 0.5).astype(int)
final_validation_score = accuracy_score(y_valid, valid_pred)

final_cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=0,
    random_state=RANDOM_STATE
)

final_lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE
)

final_cat_model.fit(X_cb, y, cat_features=cat_cols)
final_lgb_model.fit(X_lgb, y)

cat_test_proba = final_cat_model.predict_proba(X_test_cb)[:, 1]
lgb_test_proba = final_lgb_model.predict_proba(X_test_lgb)[:, 1]
ensemble_test_proba = 0.5 * cat_test_proba + 0.5 * lgb_test_proba
test_pred = (ensemble_test_proba >= 0.5).astype(int)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': np.where(test_pred == 1, 'Extrovert', 'Introvert')
})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {final_validation_score:.6f}')
