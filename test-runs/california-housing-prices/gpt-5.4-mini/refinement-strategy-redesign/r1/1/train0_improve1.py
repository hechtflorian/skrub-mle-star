
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except Exception as e:
    raise ImportError("skrub is required for this script") from e

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_dataops_pipeline(train_df, valid_df, target_col="median_house_value"):
    data = skrub.var("data", train_df)

    X = data.drop(columns=[target_col]).skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
    )

    pred_graph = X_vec.skb.apply(model, y=y)
    learner = pred_graph.skb.make_learner(fitted=True)

    valid_pred = learner.predict({"data": valid_df})
    return learner, valid_pred


def main():
    train_df, test_df = load_data()

    train_part_df, valid_part_df = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    try:
        learner, valid_pred = build_dataops_pipeline(train_part_df, valid_part_df)
        final_validation_score = rmse(valid_part_df["median_house_value"], valid_pred)
    except Exception as e:
        X_train = train_part_df.drop(columns=["median_house_value"])
        y_train = train_part_df["median_house_value"]
        X_valid = valid_part_df.drop(columns=["median_house_value"])

        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer

        fallback_model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                random_state=42,
                learning_rate=0.05,
                max_depth=6,
                max_iter=300,
                min_samples_leaf=20,
            ),
        )
        fallback_model.fit(X_train, y_train)
        valid_pred = fallback_model.predict(X_valid)
        final_validation_score = rmse(valid_part_df["median_house_value"], valid_pred)

    print(f"Final Validation Performance: {final_validation_score}")

    full_train_df = train_df.copy()
    data = skrub.var("data", full_train_df)
    X = data.drop(columns=["median_house_value"]).skb.mark_as_X()
    y = data["median_house_value"].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
    )

    pred_graph = X_vec.skb.apply(model, y=y)

    try:
        full_learner = pred_graph.skb.make_learner(fitted=True)
        test_pred = full_learner.predict({"data": test_df})
    except Exception:
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer

        X_full = full_train_df.drop(columns=["median_house_value"])
        y_full = full_train_df["median_house_value"]
        X_test = test_df.copy()

        fallback_model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                random_state=42,
                learning_rate=0.05,
                max_depth=6,
                max_iter=300,
                min_samples_leaf=20,
            ),
        )
        fallback_model.fit(X_full, y_full)
        test_pred = fallback_model.predict(X_test)

    submission = pd.DataFrame({"median_house_value": np.asarray(test_pred)})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
