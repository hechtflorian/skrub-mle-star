
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

    baseline_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        random_state=42,
    )
    alt_model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_depth=8,
        max_iter=400,
        random_state=42,
    )

    baseline_learner, baseline_rmse = fit_and_validate_variant(
        train_part, valid_part, target_col, baseline_model, "baseline"
    )
    alt_learner, alt_rmse = fit_and_validate_variant(
        train_part, valid_part, target_col, alt_model, "alt_higher_capacity"
    )

    final_validation_score = min(baseline_rmse, alt_rmse)
    print(f"Final Validation Performance: {final_validation_score}")

    if alt_rmse < baseline_rmse:
        chosen_model = alt_model
    else:
        chosen_model = baseline_model

    full_learner = train_full_model(train_df, target_col, chosen_model)
    test_pred = full_learner.predict({"data": test_df})

    os.makedirs("./final", exist_ok=True)
    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("./final/submission.csv", index=False)


if __name__ == "__main__":
    main()
