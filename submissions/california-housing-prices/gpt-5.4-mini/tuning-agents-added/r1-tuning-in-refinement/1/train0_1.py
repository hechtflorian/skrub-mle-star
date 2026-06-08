
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def make_cat_model():
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
    )


def make_lgb_model():
    return LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE)

    data = skrub.var("data", tr_df)
    X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    cat_model = make_cat_model()
    lgb_model = make_lgb_model()

    cat_learner = X.skb.apply(cat_model, y=y).skb.make_learner(fitted=True)
    lgb_learner = X.skb.apply(lgb_model, y=y).skb.make_learner(fitted=True)

    cat_val_pred = cat_learner.predict({"data": va_df})
    lgb_val_pred = lgb_learner.predict({"data": va_df})

    val_pred = 0.5 * np.asarray(cat_val_pred) + 0.5 * np.asarray(lgb_val_pred)
    final_validation_score = rmse(va_df[TARGET_COL].values, val_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    full_data = skrub.var("full_data", train_df)
    full_X = full_data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
    full_y = full_data[TARGET_COL].skb.mark_as_y()

    final_cat_model = make_cat_model()
    final_lgb_model = make_lgb_model()

    final_cat_learner = full_X.skb.apply(final_cat_model, y=full_y).skb.make_learner(fitted=True)
    final_lgb_learner = full_X.skb.apply(final_lgb_model, y=full_y).skb.make_learner(fitted=True)

    cat_test_pred = final_cat_learner.predict({"full_data": test_df})
    lgb_test_pred = final_lgb_learner.predict({"full_data": test_df})

    test_predictions = 0.5 * np.asarray(cat_test_pred) + 0.5 * np.asarray(lgb_test_pred)

    submission = pd.DataFrame({TARGET_COL: test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
