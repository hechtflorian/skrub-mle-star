
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

target_col = "count"
random_state = 42


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def add_time_features(df):
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["date"] = dt.dt.date.astype(str)
    return out


def build_feature_frame(df):
    df = add_time_features(df)
    # Minimal bug fix: exclude leakage columns that are absent in test.csv
    df = df.drop(columns=["casual", "registered"], errors="ignore")
    return df


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

train_part_feat = build_feature_frame(train_part)
valid_part_feat = build_feature_frame(valid_part)
test_df_feat = build_feature_frame(test_df)
data_full_df = build_feature_frame(train_df)

# Validation pipeline 1
data_1 = skrub.var("data", train_part_feat)
X_1 = data_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_1 = data_1[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()
predictor_1 = (
    X_1
    .skb.apply(vectorizer_1)
    .skb.apply(HistGradientBoostingRegressor(random_state=random_state), y=y_1)
)
learner_1 = predictor_1.skb.make_learner(fitted=True)
valid_p1 = np.maximum(
    np.asarray(learner_1.predict({"data": valid_part_feat}), dtype=float).ravel(), 0
)

# Validation pipeline 2
data_2 = skrub.var("data", train_part_feat)
X_2 = data_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_2 = data_2[target_col].skb.mark_as_y()

vectorizer_2 = skrub.TableVectorizer()
predictor_2 = (
    X_2
    .skb.apply(vectorizer_2)
    .skb.apply(
        RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        y=y_2,
    )
)
learner_2 = predictor_2.skb.make_learner(fitted=True)
valid_p2 = np.maximum(
    np.asarray(learner_2.predict({"data": valid_part_feat}), dtype=float).ravel(), 0
)

valid_pred = 0.5 * valid_p1 + 0.5 * valid_p2
final_validation_score = rmsle(valid_part[target_col].values, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Full-data learner 1
data_full = skrub.var("data", data_full_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

vectorizer_1_full = skrub.TableVectorizer()
predictor_1_full = (
    X_full
    .skb.apply(vectorizer_1_full)
    .skb.apply(HistGradientBoostingRegressor(random_state=random_state), y=y_full)
)
learner_1_full = predictor_1_full.skb.make_learner(fitted=True)
test_p1 = np.maximum(
    np.asarray(learner_1_full.predict({"data": test_df_feat}), dtype=float).ravel(), 0
)

# Full-data learner 2
vectorizer_2_full = skrub.TableVectorizer()
predictor_2_full = (
    X_full
    .skb.apply(vectorizer_2_full)
    .skb.apply(
        RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        y=y_full,
    )
)
learner_2_full = predictor_2_full.skb.make_learner(fitted=True)
test_p2 = np.maximum(
    np.asarray(learner_2_full.predict({"data": test_df_feat}), dtype=float).ravel(), 0
)

test_pred = np.maximum(0.5 * test_p1 + 0.5 * test_p2, 0)

submission = pd.DataFrame({
    "datetime": test_df["datetime"],
    "count": test_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(submission.head())
