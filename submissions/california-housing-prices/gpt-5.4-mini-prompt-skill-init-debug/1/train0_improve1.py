
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except Exception:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "skrub"])
    import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


def add_features(df):
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / np.maximum(df["households"], 1)
    df["bedrooms_per_room"] = df["total_bedrooms"] / np.maximum(df["total_rooms"], 1)
    df["population_per_household"] = df["population"] / np.maximum(df["households"], 1)
    df["rooms_per_population"] = df["total_rooms"] / np.maximum(df["population"], 1)
    return df


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X = skrub.X(train_df.drop(columns=["median_house_value"]))
    y = skrub.y(train_df["median_house_value"])

    # Preserve DataOps structure
    X = X.skb.apply_func(add_features)

    X_eval = X.skb.eval()
    y_eval = y.skb.eval()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_eval, y_eval, test_size=0.2, random_state=42
    )

    X_train = skrub.X(X_train)
    y_train = skrub.y(y_train)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    learner = X_train.skb.apply(model, y=y_train)
    fitted = learner.skb.make_learner()
    fitted.fit(X_train.skb.eval(), y_train.skb.eval())

    valid_pred = fitted.predict(X_valid)
    final_validation_score = mean_squared_error(y_valid, valid_pred, squared=False)
    print(f"Final Validation Performance: {final_validation_score}")

    full_X = skrub.X(add_features(train_df.drop(columns=["median_house_value"])))
    full_y = skrub.y(train_df["median_house_value"])
    full_learner = full_X.skb.apply(model, y=full_y)
    full_fitted = full_learner.skb.make_learner()
    full_fitted.fit(full_X.skb.eval(), full_y.skb.eval())

    test_pred = full_fitted.predict(add_features(test_df))

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
