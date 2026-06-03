
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
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    # Slightly stronger, still lightweight preprocessing/model choice:
    # - explicitly keep the DataOps table structure
    # - use a reusable fitted preprocessing op for CV and final fit
    # - prefer a more expressive tree model with conservative regularization
    # - keep the fallback path intact

    def make_model():
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_depth=10,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=0.1,
            max_iter=450,
            random_state=42,
        )

    # If available, reduce to meaningful column types before vectorization.
    # This keeps the DataOps pipeline focused and avoids unnecessary feature noise.
    try:
        X_for_ops = X.skb.select_dtypes(include=["number", "category", "object"])
        X_test_for_ops = test_df[X_for_ops.columns].copy()
    except Exception:
        X_for_ops = X
        X_test_for_ops = test_df.copy()

    # Reusable preprocessing op fitted on the training table, then applied consistently.
    try:
        vectorizer = skrub.TableVectorizer(
            high_cardinality="auto",
            max_onehot_levels=20,
            drop_null_columns=False,
        )
    except Exception:
        vectorizer = skrub.TableVectorizer()

    # Build a reusable transformed training representation for CV and final fit.
    X_transformed = X_for_ops.skb.apply(vectorizer)

    # Cross-validation on the transformed training data when DataOps supports it.
    try:
        cv_scores = X_transformed.skb.apply(make_model(), y=y).skb.cross_validate(
            cv=5, scoring="neg_root_mean_squared_error"
        )
        final_validation_score = float((-cv_scores["test_score"]).mean())
    except Exception:
        # Fallback manual CV using the same fitted preprocessing per fold.
        X_eval = X_for_ops.skb.eval()
        y_eval = y.skb.eval()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses = []

        for tr_idx, va_idx in kf.split(X_eval):
            X_tr, X_va = X_eval.iloc[tr_idx], X_eval.iloc[va_idx]
            y_tr, y_va = y_eval.iloc[tr_idx], y_eval.iloc[va_idx]

            fold_vectorizer = skrub.TableVectorizer(
                high_cardinality="auto",
                max_onehot_levels=20,
                drop_null_columns=False,
            )
            X_tr_t = fold_vectorizer.fit_transform(X_tr)
            X_va_t = fold_vectorizer.transform(X_va)

            fold_model = make_model()
            fold_model.fit(X_tr_t, y_tr)
            pred = fold_model.predict(X_va_t)
            rmses.append(mean_squared_error(y_va, pred, squared=False))

        final_validation_score = float(np.mean(rmses))

    # Final training on full data with the same preprocessing object.
    X_full = X_for_ops.skb.eval()
    y_full = y.skb.eval()
    X_test = X_test_for_ops.copy()

    X_full_t = vectorizer.fit_transform(X_full)
    X_test_t = vectorizer.transform(X_test)

    model = make_model()
    model.fit(X_full_t, y_full)
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
        learning_rate=0.04,
        max_depth=10,
        max_leaf_nodes=63,
        min_samples_leaf=15,
        l2_regularization=0.1,
        max_iter=450,
        random_state=42,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, va_idx in kf.split(X_train_enc):
        X_tr, X_va = X_train_enc.iloc[tr_idx], X_train_enc.iloc[va_idx]
        y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]
        model_cv = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_depth=10,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=0.1,
            max_iter=450,
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
