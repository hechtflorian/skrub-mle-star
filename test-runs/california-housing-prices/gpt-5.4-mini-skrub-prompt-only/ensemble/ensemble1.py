
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub"])
    import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def fit_and_predict(train_df, test_df, split_seed=42, bootstrap=False):
    target_col = "median_house_value"
    X_raw = train_df.drop(columns=target_col)
    y_raw = train_df[target_col]

    if bootstrap:
        rng = np.random.RandomState(split_seed)
        idx = rng.choice(len(train_df), size=len(train_df), replace=True)
        train_part = train_df.iloc[idx].reset_index(drop=True)
        X_raw = train_part.drop(columns=target_col)
        y_raw = train_part[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=split_seed
    )

    hgb_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=split_seed,
        ),
    )

    ridge_model = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=1.0, random_state=split_seed),
    )

    hgb_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)

    valid_pred_hgb = hgb_model.predict(X_valid)
    valid_pred_ridge = ridge_model.predict(X_valid)
    valid_pred = 0.8 * valid_pred_hgb + 0.2 * valid_pred_ridge

    test_pred_hgb = hgb_model.predict(test_df)
    test_pred_ridge = ridge_model.predict(test_df)
    test_pred = 0.8 * test_pred_hgb + 0.2 * test_pred_ridge

    rmse = root_mean_squared_error(y_valid, valid_pred)
    return valid_pred, test_pred, rmse


def rank_normalize(train_scores, test_scores):
    train_scores = np.asarray(train_scores)
    test_scores = np.asarray(test_scores)

    order = np.argsort(train_scores)
    sorted_train = train_scores[order]

    train_ranks = pd.Series(train_scores).rank(method="average", pct=True).to_numpy()
    test_pos = np.searchsorted(sorted_train, test_scores, side="left")
    test_ranks = test_pos / len(sorted_train)

    return train_ranks, test_ranks, sorted_train


def blend_rank_and_map_back(valid_a, test_a, valid_b, test_b, rmse_a, rmse_b):
    corr = np.corrcoef(valid_a, valid_b)[0, 1] if len(valid_a) > 1 else 1.0
    close_preds = np.allclose(valid_a, valid_b, rtol=1e-3, atol=1e-3)

    if close_preds or (np.isfinite(corr) and corr > 0.98):
        wa, wb = 0.5, 0.5
    else:
        inv_a = 1.0 / max(rmse_a, 1e-12)
        inv_b = 1.0 / max(rmse_b, 1e-12)
        s = inv_a + inv_b
        wa, wb = inv_a / s, inv_b / s

    valid_rank_a, test_rank_a, sorted_valid_a = rank_normalize(valid_a, test_a)
    valid_rank_b, test_rank_b, _ = rank_normalize(valid_b, test_b)

    blended_valid_rank = wa * valid_rank_a + wb * valid_rank_b
    blended_test_rank = wa * test_rank_a + wb * test_rank_b

    n = len(sorted_valid_a)
    blended_valid_rank = np.clip(blended_valid_rank, 0.0, 1.0)
    blended_test_rank = np.clip(blended_test_rank, 0.0, 1.0)

    valid_idx = np.minimum((blended_valid_rank * (n - 1)).astype(int), n - 1)
    test_idx = np.minimum((blended_test_rank * (n - 1)).astype(int), n - 1)

    final_valid_pred = sorted_valid_a[valid_idx]
    final_test_pred = sorted_valid_a[test_idx]

    return final_valid_pred, final_test_pred, wa, wb


def main():
    train_df, test_df = load_data()

    target_col = "median_house_value"

    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    _ = X.skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=42,
        ),
        y=y,
    )

    valid_a, test_a, rmse_a = fit_and_predict(train_df, test_df, split_seed=42, bootstrap=False)
    valid_b, test_b, rmse_b = fit_and_predict(train_df, test_df, split_seed=7, bootstrap=True)

    final_valid_pred, final_test_pred, wa, wb = blend_rank_and_map_back(
        valid_a, test_a, valid_b, test_b, rmse_a, rmse_b
    )

    y_raw = train_df[target_col].iloc[: len(final_valid_pred)].to_numpy()
    final_validation_score = root_mean_squared_error(y_raw, final_valid_pred)

    print(f"Final Validation Performance: {final_validation_score}")

    submission = pd.DataFrame({"median_house_value": final_test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
