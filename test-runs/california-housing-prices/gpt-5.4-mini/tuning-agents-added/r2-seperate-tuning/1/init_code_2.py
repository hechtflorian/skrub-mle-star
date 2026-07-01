
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Install missing dependency if needed
try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import skrub


TARGET_COL = "median_house_value"
TRAIN_PATH = os.path.join("./input", "train.csv")
TEST_PATH = os.path.join("./input", "test.csv")


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_pipeline(train_df):
    data = skrub.var("data", train_df)

    # Keep DataOps-first structure
    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    # Fast subsampling for development; keep it in the pipeline as requested
    # but do not rely on subsampled data for final validation.
    try:
        X_dev = X.skb.subsample(n=min(5000, len(train_df)))
        y_dev = y.skb.subsample(n=min(5000, len(train_df)))
        use_dev = True
    except Exception:
        X_dev = X
        y_dev = y
        use_dev = False

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    # Simple, robust preprocessing through skrub
    vectorizer = skrub.TableVectorizer()

    pred = X_dev.skb.apply(vectorizer).skb.apply(model, y=y_dev)
    return pred, use_dev


def main():
    train_df, test_df = load_data()

    train_split, valid_split = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    pred, used_subsample = build_pipeline(train_split)

    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_split})
    y_valid = valid_split[TARGET_COL].values
    final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")

    # Refit on full training data for test predictions
    full_data = skrub.var("data", train_df)
    X_full = full_data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y_full = full_data[TARGET_COL].skb.mark_as_y()

    full_model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(full_model, y=y_full)
    full_learner = full_pred.skb.make_learner(fitted=True)
    test_predictions = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
