
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_and_evaluate(train_df):
    target = "median_house_value"

    # Create validation split
    rng = np.random.RandomState(42)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx = idx[:split]
    valid_idx = idx[split:]

    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    valid_part = train_df.iloc[valid_idx].reset_index(drop=True)

    # DataOps graph: keep features and target explicitly separated
    X_train = skrub.var("data", train_part.drop(columns=[target])).skb.mark_as_X()
    y_train = skrub.var("target", train_part[target]).skb.mark_as_y()

    # Validation set does NOT contain target anymore, so do not drop it again
    valid_features = valid_part.drop(columns=[target]).copy()
    X_valid = skrub.var("data", valid_features).skb.mark_as_X()

    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(random_state=42)

    pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

    learner = pred_graph.skb.make_learner(fitted=True)

    valid_preds = learner.predict({"data": valid_features})
    final_validation_score = mean_squared_error(valid_part[target], valid_preds) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")

    return learner


def fit_full_and_predict(train_df, test_df):
    target = "median_house_value"

    X_full = skrub.var("data", train_df.drop(columns=[target])).skb.mark_as_X()
    y_full = skrub.var("target", train_df[target]).skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(random_state=42)

    pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
    learner = pred_graph.skb.make_learner(fitted=True)

    test_preds = learner.predict({"data": test_df})
    return test_preds


def main():
    train_df, test_df = load_data()

    # Validation
    _ = build_and_evaluate(train_df)

    # Train on full data and predict test
    test_preds = fit_full_and_predict(train_df, test_df)

    submission = pd.DataFrame({"median_house_value": test_preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
