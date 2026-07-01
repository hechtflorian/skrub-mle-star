
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

try:
    from catboost import CatBoostRegressor
except Exception:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostRegressor


def load_data():
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def fit_and_score(train_df, valid_df, target_col="median_house_value", model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    X_train = train_df.drop(columns=[target_col], errors="ignore")
    y_train = train_df[target_col].values
    X_valid = valid_df.drop(columns=[target_col], errors="ignore")
    y_valid = valid_df[target_col].values

    base_params = dict(
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    base_params.update(model_kwargs)

    model = CatBoostRegressor(**base_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds) ** 0.5
    return model, rmse


def build_dataops_model(train_df):
    target_col = "median_house_value"
    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    X_vec = X.skb.apply(vectorizer)

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )

    pred = X_vec.skb.apply(model, y=y)
    return pred


def main():
    train_df, test_df = load_data()

    if "median_house_value" in train_df.columns:
        train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)
    else:
        raise ValueError("Training data must contain target column 'median_house_value'.")

    # Simple ablation-style variants; fixed params are passed only once.
    variants = [
        ("baseline", {}),
        ("deeper", {"depth": 8}),
        ("more_iters", {"iterations": 3000}),
        ("deeper_more_iters", {"depth": 8, "iterations": 3000}),
    ]

    best_variant = None
    best_score = float("inf")
    best_model = None

    for variant_name, params in variants:
        model, score = fit_and_score(train_part, valid_part, model_kwargs=params)
        print(f"Ablation[{variant_name}] RMSE: {score}")
        if score < best_score:
            best_score = score
            best_variant = variant_name
            best_model = model

    print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")

    # Final DataOps-native path retained for compatibility and to preserve workflow structure.
    # Build and fit a DataOps graph; use a holdout score for validation reporting.
    data = skrub.var("data", train_part)
    X = data.drop(columns="median_house_value", errors="ignore").skb.mark_as_X()
    y = data["median_house_value"].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)
    final_model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        depth=best_model.get_params().get("depth", 6) if best_model is not None else 6,
        iterations=best_model.get_params().get("iterations", 1000) if best_model is not None else 1000,
    )
    pred = X_vec.skb.apply(final_model, y=y)

    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    final_validation_score = mean_squared_error(valid_part["median_house_value"], valid_pred) ** 0.5

    print(f"Final Validation Performance: {final_validation_score}")

    # Fit on full training data and generate submission predictions.
    full_data = skrub.var("data", train_df)
    full_X = full_data.drop(columns="median_house_value", errors="ignore").skb.mark_as_X()
    full_y = full_data["median_house_value"].skb.mark_as_y()
    full_vec = full_X.skb.apply(skrub.TableVectorizer())
    full_model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        depth=best_model.get_params().get("depth", 6) if best_model is not None else 6,
        iterations=best_model.get_params().get("iterations", 1000) if best_model is not None else 1000,
    )
    full_pred = full_vec.skb.apply(full_model, y=full_y)
    full_learner = full_pred.skb.make_learner(fitted=True)

    test_predictions = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
