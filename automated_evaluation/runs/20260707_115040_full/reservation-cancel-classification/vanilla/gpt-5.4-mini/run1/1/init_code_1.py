
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "booking_status"
id_col = "id"

X = train_df.drop(columns=[target_col])
y = train_df[target_col]
X_test = test_df.copy()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    random_seed=42,
    auto_class_weights="Balanced"
)

model.fit(
    X_tr,
    y_tr,
    cat_features=cat_cols,
    eval_set=(X_va, y_va),
    use_best_model=True
)

va_pred = model.predict_proba(X_va)[:, 1]
final_validation_score = roc_auc_score(y_va, va_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    id_col: test_df[id_col],
    target_col: test_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(submission.head())
