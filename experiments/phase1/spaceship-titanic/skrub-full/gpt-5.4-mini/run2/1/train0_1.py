
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
valid_pred_cb = np.asarray(learner_cb.predict({"data": valid_part}))
valid_pred_rf = np.asarray(learner_rf.predict({"data": valid_part}))

# Convert CatBoost probabilities / numeric outputs to boolean labels if needed
if valid_pred_cb.dtype != bool:
    if valid_pred_cb.ndim > 1:
        valid_pred_cb = valid_pred_cb[:, 1]
    valid_pred_cb = valid_pred_cb.astype(float) >= 0.5

if valid_pred_rf.dtype != bool:
    if valid_pred_rf.ndim > 1:
        valid_pred_rf = valid_pred_rf[:, 1]
    valid_pred_rf = valid_pred_rf.astype(float) >= 0.5

# Simple ensemble by averaging predicted labels
ensemble_pred = (valid_pred_cb.astype(float) + valid_pred_rf.astype(float)) / 2.0
final_pred = ensemble_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], final_pred)
print(f"Final Validation Performance: {final_validation_score}")
