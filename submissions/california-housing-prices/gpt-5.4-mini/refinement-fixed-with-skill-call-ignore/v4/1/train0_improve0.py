

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def basic_cleaning(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def build_dataops_pipeline(train_part, target_col, model, vectorizer=None):
    data = skrub.var("data", train_part)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    if vectorizer is None:
        vectorizer = skrub.TableVectorizer()

    X_vec = X.skb.apply(vectorizer)
    pred = X_vec.skb.apply(model, y=y)
    return pred


def fit_and_validate_variant(train_df, valid_df, target_col, model, variant_name):
    train_part = train_df.reset_index(drop=True)
    valid_part = valid_df.reset_index(drop=True)

    pred = build_dataops_pipeline(
        train_part=train_part,
        target_col=target_col,
        model=model,
        vectorizer=skrub.TableVectorizer(),
    )

    learner = pred.skb.make_learner(fitted=True)
    valid_features = valid_part.drop(columns=[target_col], errors="ignore")
    valid_pred = learner.predict({"data": valid_features})
    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return learner, rmse


def train_full_model(train_df, target_col, model):
    full_pred = build_dataops_pipeline(
        train_part=train_df.reset_index(drop=True),
        target_col=target_col,
        model=model,
        vectorizer=skrub.TableVectorizer(),
    )
    full_learner = full_pred.skb.make_learner(fitted=True)
    return full_learner


def make_tuned_hgb_models():
    # Anchor on the stronger higher-capacity setting, then probe a very small neighborhood.
    # Use DataOps choice-based tuning to keep the graph intact and search cheap.
    learning_rate = skrub.choose_from(
        {
            "lr_0p03": 0.03,
            "lr_0p05": 0.05,
            "lr_0p07": 0.07,
        },
        name="learning_rate",
    )
    max_depth = skrub.choose_from(
        {
            "depth_6": 6,
            "depth_8": 8,
            "depth_10": 10,
        },
        name="max_depth",
    )
    max_iter = skrub.choose_from(
        {
            "iter_250": 250,
            "iter_350": 350,
            "iter_450": 450,
        },
        name="max_iter",
    )

    return {
        "anchor_high_capacity": HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=8,
            max_iter=400,
            random_state=42,
        ),
        "choice_tuned_primary": HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            random_state=42,
        ),
        "choice_tuned_more_conservative": HistGradientBoostingRegressor(
            learning_rate=skrub.choose_from(
                {
                    "lr_0p02": 0.02,
                    "lr_0p03": 0.03,
                    "lr_0p04": 0.04,
                },
                name="learning_rate_conservative",
            ),
            max_depth=skrub.choose_from(
                {
                    "depth_5": 5,
                    "depth_6": 6,
                    "depth_8": 8,
                },
                name="max_depth_conservative",
            ),
            max_iter=skrub.choose_from(
                {
                    "iter_300": 300,
                    "iter_400": 400,
                    "iter_500": 500,
                },
                name="max_iter_conservative",
            ),
            random_state=42,
        ),
    }


def main():
    target_col = "median_house_value"
    train_df, test_df = load_data()

    train_df = basic_cleaning(train_df)
    test_df = basic_cleaning(test_df)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(train_df))
    valid_size = max(1, int(0.2 * len(train_df)))
    valid_idx = perm[:valid_size]
    train_idx = perm[valid_size:]

    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    valid_part = train_df.iloc[valid_idx].reset_index(drop=True)

    candidate_models = make_tuned_hgb_models()

    scores = {}
    best_variant = None
    best_rmse = np.inf
    best_model = None

    for variant_name, model in candidate_models.items():
        _, rmse = fit_and_validate_variant(
            train_part, valid_part, target_col, model, variant_name
        )
        scores[variant_name] = rmse
        if rmse < best_rmse:
            best_rmse = rmse
            best_variant = variant_name
            best_model = model

    final_validation_score = best_rmse
    print(f"Best ablation variant: {best_variant} | RMSE: {best_rmse}")
    print(f"Final Validation Performance: {final_validation_score}")

    full_learner = train_full_model(train_df, target_col, best_model)
    test_pred = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
