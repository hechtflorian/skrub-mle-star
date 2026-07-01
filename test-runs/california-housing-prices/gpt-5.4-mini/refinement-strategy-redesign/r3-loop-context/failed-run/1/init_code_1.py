
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_and_fit(train_df, test_df):
    target_col = "median_house_value"

    # DataOps-native feature/target marking
    data = skrub.var("data", train_df)

    # IMPORTANT FIX:
    # Only the training dataframe has the target column. Use errors="ignore"
    # so this remains safe even if the same logic is reused on test data.
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Keep the DataOps pipeline structure intact
    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(random_state=42)

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)

    # Create a holdout validation split for reporting
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    train_split = train_df.iloc[train_idx].copy()
    valid_split = train_df.iloc[valid_idx].copy()

    train_data = skrub.var("data", train_split)
    X_train = train_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_train = train_data[target_col].skb.mark_as_y()

    fitted_pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = fitted_pred.skb.make_learner(fitted=True)

    # Validation performance
    X_valid = valid_split.drop(columns=[target_col], errors="ignore")
    y_valid = valid_split[target_col].values
    valid_preds = learner.predict({"data": X_valid})
    final_validation_score = mean_squared_error(y_valid, valid_preds) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full training data and predict test data
    full_pred = pred
    full_learner = full_pred.skb.make_learner(fitted=True)

    test_preds = full_learner.predict({"data": test_df})
    return test_preds


def main():
    train_df, test_df = load_data()
    preds = build_and_fit(train_df, test_df)

    submission = pd.DataFrame({"median_house_value": preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
