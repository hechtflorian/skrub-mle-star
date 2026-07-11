
import os
import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def ensure_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
        from catboost import CatBoostClassifier
        return CatBoostClassifier

CatBoostClassifier = ensure_catboost()

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

try:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_valid_cat = encoder.transform(X_valid[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])
X_cat = encoder.transform(X[cat_cols])

encoded_cat_cols = encoder.get_feature_names_out(cat_cols)

X_train_enc = pd.concat(
    [
        X_train.drop(columns=cat_cols).reset_index(drop=True),
        pd.DataFrame(X_train_cat, columns=encoded_cat_cols).reset_index(drop=True)
    ],
    axis=1
)

X_valid_enc = pd.concat(
    [
        X_valid.drop(columns=cat_cols).reset_index(drop=True),
        pd.DataFrame(X_valid_cat, columns=encoded_cat_cols).reset_index(drop=True)
    ],
    axis=1
)

X_test_enc = pd.concat(
    [
        X_test.drop(columns=cat_cols).reset_index(drop=True),
        pd.DataFrame(X_test_cat, columns=encoded_cat_cols).reset_index(drop=True)
    ],
    axis=1
)

X_enc = pd.concat(
    [
        X.drop(columns=cat_cols).reset_index(drop=True),
        pd.DataFrame(X_cat, columns=encoded_cat_cols).reset_index(drop=True)
    ],
    axis=1
)

candidate_params = [
    {'depth': 6, 'learning_rate': 0.03},
    {'depth': 8, 'learning_rate': 0.03},
    {'depth': 6, 'learning_rate': 0.05}
]

best_model = None
best_score = -1
best_params = None

for params in candidate_params:
    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='Accuracy',
        iterations=2000,
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train_enc,
        y_train,
        eval_set=(X_valid_enc, y_valid),
        use_best_model=True,
        early_stopping_rounds=200
    )

    valid_pred = model.predict(X_valid_enc).ravel()
    score = accuracy_score(y_valid, valid_pred)
    print(f"Validation Accuracy (depth={params['depth']}, lr={params['learning_rate']}): {score}")

    if score > best_score:
        best_score = score
        best_model = model
        best_params = params

final_validation_score = best_score
print(f'Final Validation Performance: {final_validation_score}')
print(f"Best Params: {best_params}")

final_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    iterations=2000,
    learning_rate=best_params['learning_rate'],
    depth=best_params['depth'],
    random_seed=42,
    verbose=200
)

final_model.fit(X_enc, y)

test_pred = final_model.predict(X_test_enc).ravel()

submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred
})
submission.to_csv('submission.csv', index=False)
print("Saved submission to submission.csv")
