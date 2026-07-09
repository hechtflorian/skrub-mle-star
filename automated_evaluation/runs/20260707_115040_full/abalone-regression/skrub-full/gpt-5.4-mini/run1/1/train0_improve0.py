
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")

target_col = "Rings"
random_state = 42

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(
        np.sqrt(
            np.mean(
                (
                    np.log1p(np.maximum(y_pred, 0))
                    - np.log1p(np.maximum(y_true, 0))
                ) ** 2
            )
        )
    )


import numpy as np
import skrub
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_guarded_ratios(df):
    out = df.copy()
    # Keep the feature-engineering block bounded and only add ratios when the
    # likely size/weight columns are available.
    ratio_specs = [
        ("Weight", "Length", "Weight_per_Length"),
        ("Weight", "Height", "Weight_per_Height"),
        ("Length", "Height", "Length_per_Height"),
    ]
    for num_col, den_col, out_col in ratio_specs:
        if num_col in out.columns and den_col in out.columns:
            denom = out[den_col].replace(0, np.nan)
            out[out_col] = (
                (out[num_col] / denom)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
    return out


data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_guarded_ratios)

# Explicitly drop id while keeping Length intact.
X_train = (
    data_train.drop(columns=[target_col, "id"], errors="ignore")
    .skb.mark_as_X()
)
y_train = data_train[target_col].skb.mark_as_y()

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    random_seed=random_state,
    verbose=0,
    subsample=0.8,
)

pred_cat = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(cat_model, y=y_train)
learner_cat = pred_cat.skb.make_learner(fitted=True)

valid_pred = np.asarray(
    learner_cat.predict({"data": valid_part}),
    dtype=float,
).ravel()

final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

y_true = valid_part[target_col].to_numpy()

final_validation_score = rmsle(y_true, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
