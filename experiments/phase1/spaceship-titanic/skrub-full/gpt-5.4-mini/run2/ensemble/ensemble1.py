
import os
import sys
import subprocess

# Fix missing dependency with the smallest possible change.
try:
    from catboost import CatBoostClassifier
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

# Load data
train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"

# Holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph for the main workflow
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Model 1: CatBoost + TableVectorizer
vectorizer_cb = skrub.TableVectorizer()
clf_cb = CatBoostClassifier(verbose=0, random_seed=42)
pred_cb = X_train.skb.apply(vectorizer_cb).skb.apply(clf_cb, y=y_train)

# Model 2: RandomForest + explicit preprocessing
feature_df = train_part.drop(columns=target_col, errors="ignore")
numeric_features = [
    c for c in feature_df.columns
    if pd.api.types.is_numeric_dtype(feature_df[c])
]
categorical_features = [
    c for c in feature_df.columns
    if c not in numeric_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

model_rf = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42, n_estimators=300),
)

pred_rf = X_train.skb.apply(model_rf, y=y_train)

# Fit both models on the holdout training partition
learner_cb = pred_cb.skb.make_learner(fitted=True)
learner_rf = pred_rf.skb.make_learner(fitted=True)

# Predict on validation set
valid_raw_cb = np.asarray(learner_cb.predict({"data": valid_part}))
valid_raw_rf = np.asarray(learner_rf.predict({"data": valid_part}))

def extract_continuous_score(pred):
    pred = np.asarray(pred)
    if pred.ndim > 1:
        # Prefer positive-class probability when available
        if pred.shape[1] >= 2:
            pred = pred[:, 1]
        else:
            pred = pred.ravel()
    if pred.dtype == bool:
        return pred.astype(float)
    pred = pred.astype(float)
    # If predictions look like hard labels in {0,1}, keep them as continuous scores
    return pred

def calibrate_scores_isotonic(scores, y_true):
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    uniq = np.unique(scores)
    if len(uniq) < 2:
        return np.full_like(scores, fill_value=float(np.mean(y_true)), dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y_true)
    return np.asarray(iso.transform(scores), dtype=float)

valid_scores_cb = extract_continuous_score(valid_raw_cb)
valid_scores_rf = extract_continuous_score(valid_raw_rf)

# Light monotonic calibration on the holdout
valid_cal_cb = calibrate_scores_isotonic(valid_scores_cb, valid_part[target_col].astype(int).values)
valid_cal_rf = calibrate_scores_isotonic(valid_scores_rf, valid_part[target_col].astype(int).values)

# Disagreement-aware fusion:
# - average calibrated scores by default
# - on strong disagreement, choose the more confident calibrated score
ens_score = 0.5 * valid_cal_cb + 0.5 * valid_cal_rf

# Hard predictions from calibrated scores for agreement/disagreement logic
pred_cb_bool = valid_cal_cb >= 0.5
pred_rf_bool = valid_cal_rf >= 0.5
disagree = pred_cb_bool != pred_rf_bool

conf_cb = np.abs(valid_cal_cb - 0.5)
conf_rf = np.abs(valid_cal_rf - 0.5)

# Only override average when disagreement is strong and one model is clearly more confident
conf_gap = np.abs(conf_cb - conf_rf)
high_disagreement = disagree & (conf_gap > 0.05)

choose_cb = high_disagreement & (conf_cb > conf_rf)
choose_rf = high_disagreement & (conf_rf > conf_cb)

ens_score[choose_cb] = valid_cal_cb[choose_cb]
ens_score[choose_rf] = valid_cal_rf[choose_rf]

# Learn the blend threshold on the validation split with a tiny sweep
threshold_grid = np.arange(0.40, 0.601, 0.01)
best_threshold = 0.5
best_acc = -1.0
y_valid = valid_part[target_col].astype(bool).values

for thr in threshold_grid:
    pred = ens_score >= thr
    acc = accuracy_score(y_valid, pred)
    if acc > best_acc:
        best_acc = acc
        best_threshold = thr

final_pred = ens_score >= best_threshold
final_validation_score = accuracy_score(valid_part[target_col], final_pred)
print(f"Final Validation Performance: {final_validation_score}")
