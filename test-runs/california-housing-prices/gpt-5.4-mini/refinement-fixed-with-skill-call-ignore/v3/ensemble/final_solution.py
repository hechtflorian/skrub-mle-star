
import os
import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


INPUT_DIR = "./input"
FINAL_DIR = "./final"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_and_evaluate(train_df):
    train_df = train_df.copy()

    data = skrub.var("data", train_df).skb.subsample(n=min(5000, len(train_df)))

    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=6,
        max_iter=250,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)

    learner = pred.skb.make_learner(fitted=True)

    idx = np.arange(len(train_df))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)

    train_split = train_df.iloc[tr_idx].reset_index(drop=True)
    valid_split = train_df.iloc[va_idx].reset_index(drop=True)

    train_data = skrub.var("data", train_split).skb.subsample(n=min(5000, len(train_split)))
    X_train = train_data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y_train = train_data[TARGET_COL].skb.mark_as_y()

    train_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y_train,
    )
    fitted_learner = train_pred.skb.make_learner(fitted=True)

    valid_env = {"data": valid_split}
    valid_preds = fitted_learner.predict(valid_env)
    final_validation_score = mean_squared_error(valid_split[TARGET_COL], valid_preds) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    return learner


def fit_full_and_predict(train_df, test_df):
    data = skrub.var("data", train_df).skb.subsample(n=min(5000, len(train_df)))
    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y,
    )
    learner = pred.skb.make_learner(fitted=True)
    test_preds = learner.predict({"data": test_df})
    return test_preds


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)

    train_df, test_df = load_data()
    build_and_evaluate(train_df)
    test_preds = fit_full_and_predict(train_df, test_df)

    submission = pd.DataFrame({TARGET_COL: test_preds})
    submission_path = os.path.join(FINAL_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(submission.head())


if __name__ == "__main__":
    main()
