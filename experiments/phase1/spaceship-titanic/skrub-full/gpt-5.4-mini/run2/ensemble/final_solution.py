
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
valid_raw_cb = np.asarray(learner_cb.predict({"data": valid_part}))
valid_raw_rf = np.asarray(learner_rf.predict({"data": valid_part}))

def extract_continuous_score(pred):
    pred = np.asarray(pred)
    if pred.ndim > 1:
        if pred.shape[1] >= 2:
            pred = pred[:, 1]
        else:
            pred = pred.ravel()
    if pred.dtype == bool:
        return pred.astype(float)
    return pred.astype(float)

valid_scores_cb = extract_continuous_score(valid_raw_cb)
valid_scores_rf = extract_continuous_score(valid_raw_rf)

# Simple probability fusion
ens_score = 0.5 * valid_scores_cb + 0.5 * valid_scores_rf

pred_cb_bool = valid_scores_cb >= 0.5
pred_rf_bool = valid_scores_rf >= 0.5
disagree = pred_cb_bool != pred_rf_bool

# Slight trust to CatBoost on strong disagreement
strong_disagree = disagree & (np.abs(valid_scores_cb - valid_scores_rf) > 0.35)
ens_score = np.where(strong_disagree, valid_scores_cb, ens_score)

final_pred = ens_score >= 0.5
final_validation_score = accuracy_score(valid_part[target_col], final_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage only: refit on full training data, then predict test
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_cb = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(
    CatBoostClassifier(verbose=0, random_seed=42),
    y=y_full,
)

feature_df_full = train_df.drop(columns=target_col, errors="ignore")
numeric_features_full = [
    c for c in feature_df_full.columns
    if pd.api.types.is_numeric_dtype(feature_df_full[c])
]
categorical_features_full = [
    c for c in feature_df_full.columns
    if c not in numeric_features_full
]

preprocessor_full = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features_full),
        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical_features_full,
        ),
    ],
    remainder="drop",
)

full_model_rf = make_pipeline(
    preprocessor_full,
    RandomForestClassifier(random_state=42, n_estimators=300),
)

full_pred_rf = X_full.skb.apply(full_model_rf, y=y_full)

full_learner_cb = full_pred_cb.skb.make_learner(fitted=True)
full_learner_rf = full_pred_rf.skb.make_learner(fitted=True)

test_raw_cb = np.asarray(full_learner_cb.predict({"data": test_df}))
test_raw_rf = np.asarray(full_learner_rf.predict({"data": test_df}))

test_scores_cb = extract_continuous_score(test_raw_cb)
test_scores_rf = extract_continuous_score(test_raw_rf)

test_ens_score = 0.5 * test_scores_cb + 0.5 * test_scores_rf
test_pred_cb_bool = test_scores_cb >= 0.5
test_pred_rf_bool = test_scores_rf >= 0.5
test_disagree = test_pred_cb_bool != test_pred_rf_bool
test_strong_disagree = test_disagree & (np.abs(test_scores_cb - test_scores_rf) > 0.35)
test_ens_score = np.where(test_strong_disagree, test_scores_cb, test_ens_score)

test_final_pred = test_ens_score >= 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_final_pred.astype(bool),
    }
)

os.makedirs("./final", exist_ok=True)
submission.to_csv("./final/submission.csv", index=False)
