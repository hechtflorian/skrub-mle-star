
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"


@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    if "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        if "total_rooms" in out.columns:
            out["rooms_per_household"] = (
                out["total_rooms"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "population" in out.columns:
            out["people_per_household"] = (
                out["population"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def fit_and_predict(split_seed):
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=split_seed
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    data_train = skrub.var("data", train_part)
    data_train = data_train.skb.apply_func(add_ratio_features)

    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    model = CatBoostRegressor(
        iterations=5000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200,
        allow_writing_files=False,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    val_learner = pred.skb.make_learner(fitted=True)

    valid_pred = val_learner.predict({"data": valid_part})
    test_pred = val_learner.predict({"data": test_df})

    valid_rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    return valid_part, valid_pred, test_pred, valid_rmse


# Two-seed same-model ensemble
valid_part_a, valid_pred_a, test_pred_a, rmse_a = fit_and_predict(42)
valid_part_b, valid_pred_b, test_pred_b, rmse_b = fit_and_predict(123)

# Use the same validation rows for both runs to calibrate the scalar blend
# If the split row sets differ, compare on each own fold quality and fallback to inverse-RMSE weights.
if len(valid_pred_a) == len(valid_pred_b) and np.array_equal(valid_part_a.index.values, valid_part_b.index.values):
    # One-parameter linear fit on the validation set: alpha * pred_a + (1 - alpha) * pred_b
    diff = valid_pred_a - valid_pred_b
    denom = np.dot(diff, diff)
    if denom > 1e-12:
        alpha = np.dot(valid_part_a[target_col].values - valid_pred_b, diff) / denom
    else:
        alpha = 0.5
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended_valid_pred = alpha * valid_pred_a + (1.0 - alpha) * valid_pred_b
    blended_test_pred = alpha * test_pred_a + (1.0 - alpha) * test_pred_b
    final_validation_score = mean_squared_error(valid_part_a[target_col], blended_valid_pred) ** 0.5
else:
    # If the validation splits are not identical, use an RMSE-based fixed blend
    w_a = 1.0 / max(rmse_a, 1e-12)
    w_b = 1.0 / max(rmse_b, 1e-12)
    alpha = float(w_a / (w_a + w_b))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended_valid_pred = alpha * valid_pred_a + (1.0 - alpha) * valid_pred_b
    final_validation_score = mean_squared_error(valid_part_a[target_col], blended_valid_pred) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")
