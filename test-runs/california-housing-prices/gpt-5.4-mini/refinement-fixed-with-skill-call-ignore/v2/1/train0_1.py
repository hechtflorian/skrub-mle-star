
import os
import warnings
import numpy as np
import pandas as pd
import skrub

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "median_house_value"


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_dataops_learner(train_df, use_subsample=True):
    data = skrub.var("data", train_df)

    if use_subsample and len(train_df) > 5000:
        data = data.skb.subsample(n=5000)

    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=200,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred = X_vec.skb.apply(model, y=y)
    learner = pred.skb.make_learner(fitted=False)
    learner.fit({"data": train_df})
    return learner


def build_and_evaluate(train_df):
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_STATE
    )

    learner_a = build_dataops_learner(train_split, use_subsample=True)
    pred_a = learner_a.predict({"data": val_split})
    score_a = rmse(val_split[TARGET], pred_a)

    # Reference-inspired second model with different hyperparameters.
    data = skrub.var("data", train_split)
    if len(train_split) > 5000:
        data = data.skb.subsample(n=5000)

    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model_b = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred_b = X_vec.skb.apply(model_b, y=y)
    learner_b = pred_b.skb.make_learner(fitted=False)
    learner_b.fit({"data": train_split})
    pred_b_val = learner_b.predict({"data": val_split})
    score_b = rmse(val_split[TARGET], pred_b_val)

    if score_b < score_a:
        final_validation_score = score_b
    else:
        final_validation_score = score_a

    print(f"Final Validation Performance: {final_validation_score}")
    return learner_a, learner_b, final_validation_score


def fit_full_model(train_df):
    data = skrub.var("data", train_df)
    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model_1 = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=200,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    model_2 = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    pred_1 = X_vec.skb.apply(model_1, y=y)
    pred_2 = X_vec.skb.apply(model_2, y=y)

    learner_1 = pred_1.skb.make_learner(fitted=False)
    learner_2 = pred_2.skb.make_learner(fitted=False)

    learner_1.fit({"data": train_df})
    learner_2.fit({"data": train_df})
    return learner_1, learner_2


def main():
    train_df, test_df = load_data()

    _, _, final_validation_score = build_and_evaluate(train_df)

    learner_1, learner_2 = fit_full_model(train_df)
    test_pred_1 = learner_1.predict({"data": test_df})
    test_pred_2 = learner_2.predict({"data": test_df})

    test_preds = 0.5 * np.asarray(test_pred_1) + 0.5 * np.asarray(test_pred_2)

    submission = pd.DataFrame({"median_house_value": test_preds})
    submission.to_csv("submission.csv", index=False)

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
