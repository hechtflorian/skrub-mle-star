
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "NObeyesdad"
y = train[target]
X = train.drop(columns=[target])

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=2000,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=200
)

model.fit(
    X_tr,
    y_tr,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    use_best_model=True
)

val_pred = model.predict(X_val).ravel()
final_validation_score = accuracy_score(y_val, val_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_pred = model.predict(test).ravel()
submission = pd.DataFrame({
    "id": test["id"],
    "NObeyesdad": test_pred
})
submission.to_csv("submission.csv", index=False)
