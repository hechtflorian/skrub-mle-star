
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import skrub

import lightgbm as lgb

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBRegressor


TARGET_COL = "median_house_value"
TRAIN_PATH = os.path.join("./input", "train.csv")
TEST_PATH = os.path.join("./input", "test.csv")


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_dataops_learner(train_df, model, vectorizer):
    data = skrub.var("data", train_df)
    X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()
    X_vec = X.skb.apply(vectorizer)
    predictor = X_vec.skb.apply(model, y=y)
    learner = predictor.skb.make_learner(fitted=True)
    return learner


def main():
    train_df, test_df = load_data()

    train_split, valid_split = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    lgb_vectorizer = skrub.TableVectorizer()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgb_learner = build_dataops_learner(train_split, lgb_model, lgb_vectorizer)
    valid_pred_lgb = lgb_learner.predict({"data": valid_split})

    xgb_vectorizer = skrub.TableVectorizer()
    xgb_model = XGBRegressor(
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
    xgb_learner = build_dataops_learner(train_split, xgb_model, xgb_vectorizer)
    valid_pred_xgb = xgb_learner.predict({"data": valid_split})

    valid_pred_ensemble = 0.5 * valid_pred_lgb + 0.5 * valid_pred_xgb
    y_valid = valid_split[TARGET_COL].values
    final_validation_score = mean_squared_error(y_valid, valid_pred_ensemble) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")

    full_lgb_learner = build_dataops_learner(
        train_df,
        lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        skrub.TableVectorizer(),
    )
    full_xgb_learner = build_dataops_learner(
        train_df,
        XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
        skrub.TableVectorizer(),
    )

    test_pred_lgb = full_lgb_learner.predict({"data": test_df})
    test_pred_xgb = full_xgb_learner.predict({"data": test_df})
    test_predictions = 0.5 * test_pred_lgb + 0.5 * test_pred_xgb

    submission = pd.DataFrame({TARGET_COL: test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
