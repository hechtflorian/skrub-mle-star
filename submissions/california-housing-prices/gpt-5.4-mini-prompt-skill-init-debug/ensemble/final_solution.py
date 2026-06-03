
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
FINAL_DIR = "./final"
os.makedirs(FINAL_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ---------------------------
# Solution 1 pipeline
# ---------------------------
data1 = skrub.var("data1", train_df)

X1 = data1.skb.drop(TARGET_COL).skb.mark_as_X()
y1 = data1[TARGET_COL].skb.mark_as_y()

try:
    vectorizer1 = skrub.TableVectorizer()
    model1 = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )

    preds1 = X1.skb.apply(vectorizer1).skb.apply(model1, y=y1)

    try:
        cv_scores1 = preds1.skb.cross_validate(cv=5, scoring="neg_root_mean_squared_error")
        final_validation_score1 = float((-cv_scores1["test_score"]).mean())
    except Exception:
        X1_eval = X1.skb.eval()
        y1_eval = y1.skb.eval()
        kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses1 = []
        for tr_idx, va_idx in kf1.split(X1_eval):
            X_tr, X_va = X1_eval.iloc[tr_idx], X1_eval.iloc[va_idx]
            y_tr, y_va = y1_eval.iloc[tr_idx], y1_eval.iloc[va_idx]
            pipe_model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=8,
                max_iter=300,
                min_samples_leaf=20,
                random_state=42,
            )
            X_tr_t = vectorizer1.fit_transform(X_tr)
            X_va_t = vectorizer1.transform(X_va)
            pipe_model.fit(X_tr_t, y_tr)
            pred = pipe_model.predict(X_va_t)
            rmses1.append(mean_squared_error(y_va, pred, squared=False))
        final_validation_score1 = float(np.mean(rmses1))

    X1_full = X1.skb.eval()
    y1_full = y1.skb.eval()
    X_test1 = test_df.copy()

    X_train_t1 = vectorizer1.fit_transform(X1_full)
    X_test_t1 = vectorizer1.transform(X_test1)
    model1.fit(X_train_t1, y1_full)
    test_pred1 = model1.predict(X_test_t1)

    train_pred1 = model1.predict(X_train_t1)
    mu1 = float(np.mean(train_pred1))
    sigma1 = float(np.std(train_pred1) + 1e-12)

except Exception:
    X1_full = train_df.drop(columns=[TARGET_COL])
    y1_full = train_df[TARGET_COL]
    X_test1 = test_df.copy()

    X_all1 = pd.concat([X1_full, X_test1], axis=0, ignore_index=True)
    X_all1 = pd.get_dummies(X_all1, dummy_na=True)
    X_train_enc1 = X_all1.iloc[: len(X1_full)]
    X_test_enc1 = X_all1.iloc[len(X1_full):]

    model1 = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )
    kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses1 = []
    for tr_idx, va_idx in kf1.split(X_train_enc1):
        X_tr, X_va = X_train_enc1.iloc[tr_idx], X_train_enc1.iloc[va_idx]
        y_tr, y_va = y1_full.iloc[tr_idx], y1_full.iloc[va_idx]
        model_cv = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            min_samples_leaf=20,
            random_state=42,
        )
        model_cv.fit(X_tr, y_tr)
        pred = model_cv.predict(X_va)
        rmses1.append(mean_squared_error(y_va, pred, squared=False))
    final_validation_score1 = float(np.mean(rmses1))

    model1.fit(X_train_enc1, y1_full)
    test_pred1 = model1.predict(X_test_enc1)
    train_pred1 = model1.predict(X_train_enc1)
    mu1 = float(np.mean(train_pred1))
    sigma1 = float(np.std(train_pred1) + 1e-12)

# ---------------------------
# Solution 2 pipeline
# ---------------------------
data2 = skrub.var("data2", train_df)

X2 = data2.skb.drop(TARGET_COL).skb.mark_as_X()
y2 = data2[TARGET_COL].skb.mark_as_y()

try:
    vectorizer2 = skrub.TableVectorizer()
    model2 = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_depth=10,
        max_iter=400,
        min_samples_leaf=15,
        random_state=42,
    )

    preds2 = X2.skb.apply(vectorizer2).skb.apply(model2, y=y2)

    try:
        cv_scores2 = preds2.skb.cross_validate(cv=5, scoring="neg_root_mean_squared_error")
        final_validation_score2 = float((-cv_scores2["test_score"]).mean())
    except Exception:
        X2_eval = X2.skb.eval()
        y2_eval = y2.skb.eval()
        kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses2 = []
        for tr_idx, va_idx in kf2.split(X2_eval):
            X_tr, X_va = X2_eval.iloc[tr_idx], X2_eval.iloc[va_idx]
            y_tr, y_va = y2_eval.iloc[tr_idx], y2_eval.iloc[va_idx]
            pipe_model = HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=10,
                max_iter=400,
                min_samples_leaf=15,
                random_state=42,
            )
            X_tr_t = vectorizer2.fit_transform(X_tr)
            X_va_t = vectorizer2.transform(X_va)
            pipe_model.fit(X_tr_t, y_tr)
            pred = pipe_model.predict(X_va_t)
            rmses2.append(mean_squared_error(y_va, pred, squared=False))
        final_validation_score2 = float(np.mean(rmses2))

    X2_full = X2.skb.eval()
    y2_full = y2.skb.eval()
    X_test2 = test_df.copy()

    X_train_t2 = vectorizer2.fit_transform(X2_full)
    X_test_t2 = vectorizer2.transform(X_test2)
    model2.fit(X_train_t2, y2_full)
    test_pred2 = model2.predict(X_test_t2)

    train_pred2 = model2.predict(X_train_t2)
    mu2 = float(np.mean(train_pred2))
    sigma2 = float(np.std(train_pred2) + 1e-12)

except Exception:
    X2_full = train_df.drop(columns=[TARGET_COL])
    y2_full = train_df[TARGET_COL]
    X_test2 = test_df.copy()

    X_all2 = pd.concat([X2_full, X_test2], axis=0, ignore_index=True)
    X_all2 = pd.get_dummies(X_all2, dummy_na=True)
    X_train_enc2 = X_all2.iloc[: len(X2_full)]
    X_test_enc2 = X_all2.iloc[len(X2_full):]

    model2 = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_depth=10,
        max_iter=400,
        min_samples_leaf=15,
        random_state=42,
    )
    kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses2 = []
    for tr_idx, va_idx in kf2.split(X_train_enc2):
        X_tr, X_va = X_train_enc2.iloc[tr_idx], X_train_enc2.iloc[va_idx]
        y_tr, y_va = y2_full.iloc[tr_idx], y2_full.iloc[va_idx]
        model_cv = HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=10,
            max_iter=400,
            min_samples_leaf=15,
            random_state=42,
        )
        model_cv.fit(X_tr, y_tr)
        pred = model_cv.predict(X_va)
        rmses2.append(mean_squared_error(y_va, pred, squared=False))
    final_validation_score2 = float(np.mean(rmses2))

    model2.fit(X_train_enc2, y2_full)
    test_pred2 = model2.predict(X_test_enc2)
    train_pred2 = model2.predict(X_train_enc2)
    mu2 = float(np.mean(train_pred2))
    sigma2 = float(np.std(train_pred2) + 1e-12)

# ---------------------------
# Two-stage ensemble blend
# ---------------------------
score1 = float(final_validation_score1)
score2 = float(final_validation_score2)
eps = 1e-12
w1 = 1.0 / max(score1, eps)
w2 = 1.0 / max(score2, eps)
wsum = w1 + w2
w1 /= wsum
w2 /= wsum

z1 = (np.asarray(test_pred1) - mu1) / (sigma1 + eps)
z2 = (np.asarray(test_pred2) - mu2) / (sigma2 + eps)

z_blend = w1 * z1 + w2 * z2
blend_mu = w1 * mu1 + w2 * mu2
blend_sigma = w1 * sigma1 + w2 * sigma2
test_pred = z_blend * blend_sigma + blend_mu

final_validation_score = float(w1 * score1 + w2 * score2)
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv(os.path.join(FINAL_DIR, "submission.csv"), index=False)
print(submission.head().to_string(index=False))
