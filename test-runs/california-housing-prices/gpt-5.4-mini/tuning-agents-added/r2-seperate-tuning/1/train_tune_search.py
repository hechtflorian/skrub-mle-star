
import os
import json
import warnings

import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

TRAIN_PATH = os.path.join("./input", "train.csv")
TEST_PATH = os.path.join("./input", "test.csv")
TARGET = "median_house_value"


def build_graph(train_df: pd.DataFrame):
    data = skrub.var("data", train_df)
    X = data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=skrub.choose_float(0.01, 0.05, name="learning_rate"),
        num_leaves=skrub.choose_int(48, 96, name="num_leaves"),
        subsample=skrub.choose_float(0.7, 0.9, name="subsample"),
        colsample_bytree=skrub.choose_float(0.7, 0.9, name="colsample_bytree"),
        random_state=42,
    )

    pred = X_vec.skb.apply(model, y=y)
    return pred


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)

    rng = np.random.RandomState(42)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    train_part = train_df.iloc[tr_idx].reset_index(drop=True)
    valid_part = train_df.iloc[va_idx].reset_index(drop=True)

    pred = build_graph(train_part)
    search = pred.skb.make_randomized_search(
        n_iter=4,
        n_jobs=2,
        random_state=42,
        fitted=True,
    )

    valid_pred = search.best_learner_.predict({"data": valid_part})
    final_validation_score = mean_squared_error(valid_part[TARGET].values, valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    best_params = search.best_params_
    print(f"TUNING_BEST_PARAMS: {json.dumps(best_params)}")

    full_pred = build_graph(train_df)
    full_search = full_pred.skb.make_randomized_search(
        n_iter=4,
        n_jobs=2,
        random_state=42,
        fitted=True,
    )
    test_pred = full_search.best_learner_.predict({"data": test_df})

    submission = pd.DataFrame({TARGET: test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
