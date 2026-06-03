
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")


def build_learner(train_df: pd.DataFrame):
    target_col = "median_house_value"

    data = skrub.var("data", train_df)

    # Fix: do not assume the target exists at inference time.
    # Use errors="ignore" so the same graph can be applied to both train and test inputs.
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=250,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
    learner = pred.skb.make_learner(fitted=True)
    return learner


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target_col = "median_house_value"

    # validation split
    rng = np.random.RandomState(42)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    val_size = max(1, int(0.2 * len(train_df)))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    learner = build_learner(tr_df)

    # validation prediction uses environment dict keyed by source variable name
    val_pred = learner.predict({"data": val_df})
    final_validation_score = mean_squared_error(val_df[target_col], val_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # fit on full training data for test predictions
    full_learner = build_learner(train_df)
    test_pred = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({target_col: test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
