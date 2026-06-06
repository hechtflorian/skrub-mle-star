
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TARGET_COL = "median_house_value"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_dataops_graph(train_df):
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    data = skrub.var("data", train_df)
    X = data[feature_cols].skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    hgb = HistGradientBoostingRegressor(random_state=0)
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=0,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    X_vec = X.skb.apply(vectorizer)
    hgb_graph = X_vec.skb.apply(hgb, y=y)
    rf_graph = X_vec.skb.apply(rf, y=y)

    hgb_learner = hgb_graph.skb.make_learner(fitted=True)
    rf_learner = rf_graph.skb.make_learner(fitted=True)

    return hgb_learner, rf_learner


def evaluate_ensemble(train_df, hgb_learner, rf_learner):
    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=0)

    hgb_learner.fit({"data": train_part})
    rf_learner.fit({"data": train_part})

    valid_features = valid_part.drop(columns=[TARGET_COL], errors="ignore")
    hgb_pred = hgb_learner.predict({"data": valid_features})
    rf_pred = rf_learner.predict({"data": valid_features})

    valid_pred = 0.6 * hgb_pred + 0.4 * rf_pred
    final_validation_score = mean_squared_error(valid_part[TARGET_COL], valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    return hgb_learner, rf_learner, final_validation_score


def fit_full_and_predict(train_df, test_df):
    hgb_learner, rf_learner = build_dataops_graph(train_df)

    hgb_learner.fit({"data": train_df})
    rf_learner.fit({"data": train_df})

    test_hgb_pred = hgb_learner.predict({"data": test_df})
    test_rf_pred = rf_learner.predict({"data": test_df})

    test_pred = 0.6 * test_hgb_pred + 0.4 * test_rf_pred
    return test_pred


def main():
    train_df, test_df = load_data()
    hgb_learner, rf_learner = build_dataops_graph(train_df)
    hgb_learner, rf_learner, _ = evaluate_ensemble(train_df, hgb_learner, rf_learner)

    test_pred = fit_full_and_predict(train_df, test_df)
    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
