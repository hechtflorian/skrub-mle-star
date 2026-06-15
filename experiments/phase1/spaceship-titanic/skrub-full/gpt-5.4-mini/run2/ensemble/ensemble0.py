
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
valid_raw_cb = learner_cb.predict({"data": valid_part})
valid_raw_rf = learner_rf.predict({"data": valid_part})

# Convert outputs to positive-class probabilities or float scores
def to_positive_score(pred):
    pred = np.asarray(pred)
    if pred.ndim > 1 and pred.shape[1] > 1:
        return pred[:, 1].astype(float)
    if pred.dtype == bool:
        return pred.astype(float)
    if pred.ndim == 1:
        unique_vals = np.unique(pred[~pd.isna(pred)]) if pred.size else np.array([])
        if set(unique_vals.tolist()).issubset({0, 1, 0.0, 1.0, False, True}):
            return pred.astype(float)
        return pred.astype(float)
    return pred.astype(float).ravel()

p_cb = to_positive_score(valid_raw_cb)
p_rf = to_positive_score(valid_raw_rf)

# Individual model validation performance for weight selection
cb_pred_label = p_cb >= 0.5
rf_pred_label = p_rf >= 0.5
acc_cb = accuracy_score(valid_part[target_col], cb_pred_label)
acc_rf = accuracy_score(valid_part[target_col], rf_pred_label)

# Simple weight-selection rule with near-equal fallback
diff = abs(acc_cb - acc_rf)
if diff < 0.005:
    w_cb, w_rf = 0.5, 0.5
elif acc_cb > acc_rf:
    w_cb, w_rf = 0.55, 0.45
else:
    w_cb, w_rf = 0.45, 0.55

# Diversity-aware tie-breaker: if disagreement is strong, trust CatBoost slightly more
disagree = cb_pred_label != rf_pred_label
ens_prob = w_cb * p_cb + w_rf * p_rf
strong_disagree = disagree & (np.abs(p_cb - p_rf) > 0.35)
ens_prob = np.where(strong_disagree, p_cb, ens_prob)

final_pred = ens_prob >= 0.5
final_validation_score = accuracy_score(valid_part[target_col], final_pred)
print(f"Final Validation Performance: {final_validation_score}")
