
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import skrub

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep the DataOps structure: create a skrub var, then derive X/y with DataOps ops.
data = skrub.var("data", train_df)

# Fix: do not wrap a DataOp inside skrub.X(). Use DataOps chaining and mark_as_X / mark_as_y.
X = data.skb.drop(TARGET_COL).skb.mark_as_X()
y = data[TARGET_COL].skb.mark_as_y()

# Optional subsampling for fast iteration if available; do not remove subsampling.
try:
    X = X.skb.subsample(n=min(5000, len(train_df)), random_state=42)
    y = y.skb.subsample(n=min(5000, len(train_df)), random_state=42)
except Exception:
    pass

# Build a simple, robust preprocessing + model pipeline while keeping the DataOps workflow.
try:
    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )

    preds = X.skb.apply(vectorizer).skb.apply(model, y=y)

    # Evaluate via cross-validation using the DataOps graph when possible.
    try:
        cv_scores = preds.skb.cross_validate(cv=5, scoring="neg_root_mean_squared_error")
        final_validation_score = float((-cv_scores["test_score"]).mean())
    except Exception:
        # Fallback manual CV on evaluated pandas objects if direct DataOps CV isn't available.
        X_eval = X.skb.eval()
        y_eval = y.skb.eval()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses = []
        for tr_idx, va_idx in kf.split(X_eval):
            X_tr, X_va = X_eval.iloc[tr_idx], X_eval.iloc[va_idx]
            y_tr, y_va = y_eval.iloc[tr_idx], y_eval.iloc[va_idx]
            pipe_model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=8,
                max_iter=300,
                min_samples_leaf=20,
                random_state=42,
            )
            X_tr_t = vectorizer.fit_transform(X_tr)
            X_va_t = vectorizer.transform(X_va)
            pipe_model.fit(X_tr_t, y_tr)
            pred = pipe_model.predict(X_va_t)
            rmses.append(mean_squared_error(y_va, pred, squared=False))
        final_validation_score = float(np.mean(rmses))

    # Train final model on full data and predict test.
    X_full = X.skb.eval()
    y_full = y.skb.eval()
    X_test = test_df.copy()

    X_train_t = vectorizer.fit_transform(X_full)
    X_test_t = vectorizer.transform(X_test)
    model.fit(X_train_t, y_full)
    test_pred = model.predict(X_test_t)

except Exception:
    # Conservative fallback if some skrub/DataOps functionality is unavailable in this environment.
    X_full = train_df.drop(columns=[TARGET_COL])
    y_full = train_df[TARGET_COL]
    X_test = test_df.copy()

    # Simple numeric fill + one-hot encoding as a robust fallback.
    X_all = pd.concat([X_full, X_test], axis=0, ignore_index=True)
    X_all = pd.get_dummies(X_all, dummy_na=True)
    X_train_enc = X_all.iloc[: len(X_full)]
    X_test_enc = X_all.iloc[len(X_full) :]

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, va_idx in kf.split(X_train_enc):
        X_tr, X_va = X_train_enc.iloc[tr_idx], X_train_enc.iloc[va_idx]
        y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]
        model_cv = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            min_samples_leaf=20,
            random_state=42,
        )
        model_cv.fit(X_tr, y_tr)
        pred = model_cv.predict(X_va)
        rmses.append(mean_squared_error(y_va, pred, squared=False))
    final_validation_score = float(np.mean(rmses))

    model.fit(X_train_enc, y_full)
    test_pred = model.predict(X_test_enc)

print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
