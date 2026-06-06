
import os
import warnings
import numpy as np
import pandas as pd

import skrub
from skrub import selectors as s
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data(path):
    return pd.read_csv(path)


def make_data_features(df, drop_households=False):
    out = df.copy()
    if "rooms_per_household" not in out.columns:
        denom = out["households"].replace(0, np.nan) if "households" in out.columns else np.nan
        if "total_rooms" in out.columns and "households" in out.columns:
            out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
    if drop_households:
        out = out.drop(columns=["households"], errors="ignore")
    return out


def build_pipeline(train_df, target_col="median_house_value"):
    data = skrub.var("data", train_df)

    # Keep the original DataOps structure, but avoid double-dropping `households`.
    # The feature maker already handles the optional removal, so the later DropCols
    # call would fail when the column is absent.
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Build features in a single place.
    # We keep households by default so the engineered ratio can be created.
    # If the original ablation variant requests it, the feature function can drop it.
    @skrub.deferred
    def featurize(df):
        return make_data_features(df, drop_households=False)

    X = X.skb.apply_func(featurize)

    # Preserve DataOps pipeline structure; no redundant DropCols on households here.
    X = X.skb.apply(skrub.TableVectorizer())

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
        random_state=0,
    )

    pred = X.skb.apply(model, y=y)
    return pred


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = load_data(train_path)
    test_df = load_data(test_path)

    target_col = "median_house_value"

    # Holdout validation for a stable final metric print.
    train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

    pred = build_pipeline(train_part, target_col=target_col)
    learner = pred.skb.make_learner(fitted=True)

    valid_pred = learner.predict({"data": valid_part})
    y_valid = valid_part[target_col].to_numpy()
    final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full training data and generate submission predictions.
    full_pred = build_pipeline(train_df, target_col=target_col)
    full_learner = full_pred.skb.make_learner(fitted=True)
    test_pred = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({target_col: test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
