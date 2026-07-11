
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

train_path = os.path.join('.', 'input', 'train.csv')
test_path = os.path.join('.', 'input', 'test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = 'booking_status'
ignore = ['id', target]
features = [c for c in train.columns if c not in ignore]

X = train[features].copy()
y = train[target].copy()
X_test = test[features].copy()

cat_cols = [c for c in features if X[c].dtype == 'object']
cat_idx = [features.index(c) for c in cat_cols]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=200
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_idx,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

valid_pred = model.predict_proba(X_valid)[:, 1]
final_validation_score = roc_auc_score(y_valid, valid_pred)
print(f'Final Validation Performance: {final_validation_score}')

test_pred = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'id': test['id'],
    'booking_status': test_pred
})
submission.to_csv('submission_catboost.csv', index=False)
