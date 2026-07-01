
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE)

    # DataOps graph for validation training
    data = skrub.var("data", tr_df)
    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    learner = X.skb.apply(model, y=y).skb.make_learner(fitted=True)
    val_pred = learner.predict({"data": va_df})
    final_validation_score = rmse(va_df[TARGET_COL].values, val_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    # Final training on all training data
    full_data = skrub.var("full_data", train_df)
    full_X = full_data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    full_y = full_data[TARGET_COL].skb.mark_as_y()

    final_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    final_learner = full_X.skb.apply(final_model, y=full_y).skb.make_learner(fitted=True)
    test_predictions = final_learner.predict({"full_data": test_df})

    submission = pd.DataFrame({TARGET_COL: test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
