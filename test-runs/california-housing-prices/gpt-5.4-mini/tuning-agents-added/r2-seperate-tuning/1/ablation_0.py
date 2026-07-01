
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

import skrub
from lightgbm import LGBMRegressor


TARGET = "median_house_value"
TRAIN_PATH = os.path.join("./input", "train.csv")
TEST_PATH = os.path.join("./input", "test.csv")
SUBMISSION_PATH = "submission.csv"


def make_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "total_bedrooms" in df.columns:
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

    # Keep subsampling if present; use a simple, safe subsample only for faster validation.
    return df


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = make_feature_table(train_df)
test_df = make_feature_table(test_df)

data = skrub.var("data", train_df)
X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

# Keep the DataOps structure intact.
vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

# Bug fix: do not pass num_leaves again here, since fit_and_score/simpler_model already sets it.
model = LGBMRegressor(
    num_leaves=64,
    n_estimators=1200,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
    n_jobs=-1,
)

pred_graph = X_vec.skb.apply(model, y=y)

# Train/validation split for a direct validation score.
rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(len(idx) * 0.8)
train_idx = idx[:split]
valid_idx = idx[split:]

X_train = train_df.iloc[train_idx].drop(columns=TARGET, errors="ignore")
y_train = train_df.iloc[train_idx][TARGET].values
X_valid = train_df.iloc[valid_idx].drop(columns=TARGET, errors="ignore")
y_valid = train_df.iloc[valid_idx][TARGET].values

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    skrub.TableVectorizer(),
    LGBMRegressor(
        num_leaves=64,
        n_estimators=1200,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
    ),
)

pipeline.fit(X_train, y_train)
valid_pred = pipeline.predict(X_valid)
final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on all data and predict test.
pipeline.fit(train_df.drop(columns=TARGET, errors="ignore"), train_df[TARGET].values)
test_pred = pipeline.predict(test_df)

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)
