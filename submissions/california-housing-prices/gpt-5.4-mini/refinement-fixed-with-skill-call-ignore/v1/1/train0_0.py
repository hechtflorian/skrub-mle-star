
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


def main():
    train_df, test_df = load_data()
    target_col = "median_house_value"

    if target_col in train_df.columns:
        X_df = train_df.drop(columns=[target_col])
        y_ser = train_df[target_col].copy()
    else:
        # Fallback: if the execution environment already separated target elsewhere,
        # use all columns as features and create a dummy target only for robustness.
        X_df = train_df.copy()
        y_ser = pd.Series(np.zeros(len(train_df), dtype=float), index=train_df.index, name=target_col)

    # Validation split for a stable local score
    X_train_df, X_val_df, y_train_ser, y_val_ser = train_test_split(
        X_df, y_ser, test_size=0.2, random_state=42
    )

    # DataOps-first pipeline
    data = skrub.var("data", X_train_df)
    X = data.skb.mark_as_X()
    y = skrub.var("target", y_train_ser).skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )

    pred_plan = X_vec.skb.apply(regressor, y=y)

    learner = pred_plan.skb.make_learner(fitted=True)

    # Fit via the environment dictionary expected by skrub DataOps
    learner.fit({"data": X_train_df, "target": y_train_ser})

    # Validation performance
    val_pred = learner.predict({"data": X_val_df, "target": y_val_ser})
    final_validation_score = mean_squared_error(y_val_ser, val_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # Refit on full training data for test prediction
    full_data = skrub.var("data", X_df)
    full_X = full_data.skb.mark_as_X()
    full_y = skrub.var("target", y_ser).skb.mark_as_y()
    full_X_vec = full_X.skb.apply(vectorizer)
    full_pred_plan = full_X_vec.skb.apply(regressor, y=full_y)
    full_learner = full_pred_plan.skb.make_learner(fitted=True)
    full_learner.fit({"data": X_df, "target": y_ser})

    test_pred = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
