
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

# Ensure catboost is installed
try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

TARGET = 'NObeyesdad'
ID = 'id'

train_path = './input/train.csv'
test_path = './input/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

cat_cols = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'
]

X = train.drop(columns=[ID, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID])

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

class_names = np.sort(y.unique())

def align_proba(proba, model_classes, all_classes):
    proba_df = pd.DataFrame(proba, columns=model_classes)
    proba_df = proba_df.reindex(columns=all_classes, fill_value=0.0)
    return proba_df.values

# Faster CatBoost settings to avoid timeout
base_params = {
    'loss_function': 'MultiClass',
    'eval_metric': 'Accuracy',
    'iterations': 500,
    'learning_rate': 0.08,
    'depth': 6,
    'verbose': False,
    'od_type': 'Iter',
    'od_wait': 50,
    'thread_count': -1
}

# Validation ensemble
# Reduced number of models to avoid timeout while keeping ensemble behavior
full_model_seeds = [42]
full_valid_probas = []

for seed in full_model_seeds:
    model = CatBoostClassifier(
        **base_params,
        random_seed=seed
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )
    valid_proba = model.predict_proba(X_valid)
    valid_proba = align_proba(valid_proba, model.classes_, class_names)
    full_valid_probas.append(valid_proba)

full_valid_proba_mean = np.mean(full_valid_probas, axis=0)

# Keep subsampling/cross-validation but reduce folds for speed
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_valid_proba = np.zeros((len(X_valid), len(class_names)))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_va = y_train.iloc[va_idx]

    model = CatBoostClassifier(
        **base_params,
        random_seed=42 + fold
    )
    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    fold_valid_proba = model.predict_proba(X_valid)
    fold_valid_proba = align_proba(fold_valid_proba, model.classes_, class_names)
    cv_valid_proba += fold_valid_proba

cv_valid_proba_mean = cv_valid_proba / 3.0

ensemble_valid_proba = 0.5 * full_valid_proba_mean + 0.5 * cv_valid_proba_mean
valid_pred = class_names[np.argmax(ensemble_valid_proba, axis=1)]
final_validation_score = accuracy_score(y_valid, valid_pred)
print(f'Final Validation Performance: {final_validation_score}')

# Full-data ensemble for test prediction
full_test_probas = []
for seed in full_model_seeds:
    final_model = CatBoostClassifier(
        **base_params,
        random_seed=seed
    )
    final_model.fit(
        X,
        y,
        cat_features=cat_cols
    )
    test_proba = final_model.predict_proba(X_test)
    test_proba = align_proba(test_proba, final_model.classes_, class_names)
    full_test_probas.append(test_proba)

full_test_proba_mean = np.mean(full_test_probas, axis=0)

cv_test_proba = np.zeros((len(X_test), len(class_names)))
skf_full = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(skf_full.split(X, y), 1):
    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y.iloc[va_idx]

    model = CatBoostClassifier(
        **base_params,
        random_seed=42 + fold
    )
    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    fold_test_proba = model.predict_proba(X_test)
    fold_test_proba = align_proba(fold_test_proba, model.classes_, class_names)
    cv_test_proba += fold_test_proba

cv_test_proba_mean = cv_test_proba / 3.0

ensemble_test_proba = 0.5 * full_test_proba_mean + 0.5 * cv_test_proba_mean
test_pred = class_names[np.argmax(ensemble_test_proba, axis=1)]

submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred
})
submission.to_csv('submission.csv', index=False)
print('Saved submission.csv')
