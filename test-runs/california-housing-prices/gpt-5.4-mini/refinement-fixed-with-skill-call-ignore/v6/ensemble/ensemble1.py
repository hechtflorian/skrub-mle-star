
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


def build_and_evaluate(train_df, test_df):
    target_col = "median_house_value"

    train_df = train_df.copy()

    # Optional fast iteration subsample on the training data only if the dataset is large.
    # Keep subsampling in place as requested, but do not rely on it for final scoring.
    full_data = skrub.var("data", train_df)
    data = full_data.skb.subsample(n=min(len(train_df), 5000))

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    # Main DataOps path
    X_vec = X.skb.apply(vectorizer)
    pred = X_vec.skb.apply(
        HistGradientBoostingRegressor(random_state=0),
        y=y,
    )

    # Quick debug evaluation on subsample (kept for fast iteration)
    try:
        _ = pred.skb.cross_validate(keep_subsampling=True)
    except Exception:
        pass

    # Compile learner from the DataOps graph
    learner = pred.skb.make_learner(fitted=True)

    # Validation split for performance reporting
    train_part, val_part = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    # Rebuild on the training split using the same DataOps structure
    train_data = skrub.var("data", train_part)
    X_train = train_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_data[target_col].skb.mark_as_y()

    X_train_vec = X_train.skb.apply(skrub.TableVectorizer())
    train_pred = X_train_vec.skb.apply(
        HistGradientBoostingRegressor(random_state=0),
        y=y_train,
    )
    train_learner = train_pred.skb.make_learner(fitted=True)

    # IMPORTANT FIX:
    # The learner was built with skrub.var("data", ...), so prediction must use {"data": ...}
    val_predictions = train_learner.predict({"data": val_part})

    final_validation_score = mean_squared_error(
        val_part[target_col].to_numpy(),
        np.asarray(val_predictions),
    ) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full training data and predict test set
    full_learner = pred.skb.make_learner(fitted=True)
    test_predictions = full_learner.predict({"data": test_df})

    return test_predictions


def main():
    train_df, test_df = load_data()
    predictions = build_and_evaluate(train_df, test_df)

    submission = pd.DataFrame({"median_house_value": np.asarray(predictions)})
    submission.to_csv("submission.csv", index=False)

    # Print a preview in the required submission-like format
    print("median_house_value")
    for v in submission["median_house_value"].head(10).tolist():
        print(v)


if __name__ == "__main__":
    main()
