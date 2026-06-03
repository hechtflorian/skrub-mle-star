
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub", "-q"])
    import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_dataops_pipeline(train_df, learning_rate, max_iter):
    data = skrub.var("data", train_df)
    X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=6,
        max_iter=max_iter,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
    return pred


def get_valid_targets(valid_split):
    return np.asarray(valid_split[TARGET_COL]).ravel()


def evaluate_variant(train_split, valid_split, learning_rate, max_iter):
    train_data = skrub.var("data", train_split)
    X_train = train_data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
    y_train = train_data[TARGET_COL].skb.mark_as_y()

    pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=6,
            max_iter=max_iter,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y_train,
    )
    learner = pred.skb.make_learner(fitted=True)
    valid_preds = learner.predict({"data": valid_split})
    score = mean_squared_error(get_valid_targets(valid_split), np.asarray(valid_preds).ravel()) ** 0.5
    return score


def build_and_evaluate(train_df):
    train_df = train_df.copy()

    idx = np.arange(len(train_df))
    tr_idx, va_idx = np.random.RandomState(RANDOM_STATE).permutation(idx)[: int(0.8 * len(idx))], None
    if len(idx) > 1:
        from sklearn.model_selection import train_test_split
        tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    train_split = train_df.iloc[tr_idx].reset_index(drop=True)
    valid_split = train_df.iloc[va_idx].reset_index(drop=True)

    candidate_variants = [
        {"name": "baseline", "learning_rate": 0.08, "max_iter": 250},
        {"name": "reference_like", "learning_rate": 0.05, "max_iter": 300},
    ]

    scores = []
    for variant in candidate_variants:
        score = evaluate_variant(
            train_split=train_split,
            valid_split=valid_split,
            learning_rate=variant["learning_rate"],
            max_iter=variant["max_iter"],
        )
        scores.append((variant["name"], score))
        print(f"Ablation[{variant['name']}] RMSE: {score}")

    best_variant_name, best_score = min(scores, key=lambda x: x[1])
    print(f"Best ablation variant: {best_variant_name} | RMSE: {best_score}")

    # Fit an ensemble of two DataOps learners on the training split.
    train_data = skrub.var("data", train_split)
    X_train = train_data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
    y_train = train_data[TARGET_COL].skb.mark_as_y()

    pred_base = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y_train,
    )
    pred_ref = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y_train,
    )

    learner_base = pred_base.skb.make_learner(fitted=True)
    learner_ref = pred_ref.skb.make_learner(fitted=True)

    valid_preds_base = np.asarray(learner_base.predict({"data": valid_split})).ravel()
    valid_preds_ref = np.asarray(learner_ref.predict({"data": valid_split})).ravel()
    ensemble_valid_preds = 0.5 * valid_preds_base + 0.5 * valid_preds_ref

    final_validation_score = mean_squared_error(
        get_valid_targets(valid_split), ensemble_valid_preds
    ) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    return (learner_base, learner_ref)


def fit_full_and_predict(train_df, test_df):
    data = skrub.var("data", train_df)
    X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    pred_base = X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y,
    )
    pred_ref = X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        y=y,
    )

    learner_base = pred_base.skb.make_learner(fitted=True)
    learner_ref = pred_ref.skb.make_learner(fitted=True)

    test_preds_base = np.asarray(learner_base.predict({"data": test_df})).ravel()
    test_preds_ref = np.asarray(learner_ref.predict({"data": test_df})).ravel()
    test_preds = 0.5 * test_preds_base + 0.5 * test_preds_ref
    return test_preds


def main():
    train_df, test_df = load_data()
    build_and_evaluate(train_df)
    test_preds = fit_full_and_predict(train_df, test_df)

    submission = pd.DataFrame({TARGET_COL: np.asarray(test_preds).ravel()})
    submission.to_csv("submission.csv", index=False)
    print(submission.head())


if __name__ == "__main__":
    main()
