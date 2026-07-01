
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "median_house_value"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    eps = 1e-6
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    df["population_per_household"] = df["population"] / (df["households"] + eps)
    df["rooms_per_person"] = df["total_rooms"] / (df["population"] + eps)
    df["bedrooms_per_household"] = df["total_bedrooms"] / (df["households"] + eps)
    df["income_x_age"] = df["median_income"] * df["housing_median_age"]
    df["log_total_rooms"] = np.log1p(df["total_rooms"])
    df["log_population"] = np.log1p(df["population"])
    df["log_households"] = np.log1p(df["households"])
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)

if len(train_df) > 5000:
    preview_df = train_df.sample(n=5000, random_state=42)
else:
    preview_df = train_df.copy()

train_part, valid_part = train_test_split(preview_df, test_size=0.2, random_state=42)

# DataOps-first graph on preview split for validation
data = skrub.var("data", train_part)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

gbr = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=7,
    max_iter=400,
    min_samples_leaf=20,
    random_state=42,
)
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)

pred_gbr = X_vec.skb.apply(gbr, y=y)
pred_rf = X_vec.skb.apply(rf, y=y)

learner_gbr = pred_gbr.skb.make_learner(fitted=True)
learner_rf = pred_rf.skb.make_learner(fitted=True)

X_valid = valid_part.drop(columns=target_col, errors="ignore")
y_valid = valid_part[target_col].values

valid_env = {"data": valid_part.copy()}
pred_valid_gbr = learner_gbr.predict(valid_env)
pred_valid_rf = learner_rf.predict(valid_env)

rmse_gbr = mean_squared_error(y_valid, pred_valid_gbr) ** 0.5
rmse_rf = mean_squared_error(y_valid, pred_valid_rf) ** 0.5

if rmse_gbr <= rmse_rf:
    best_name = "HistGradientBoostingRegressor"
    final_validation_score = rmse_gbr
    best_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=7,
        max_iter=400,
        min_samples_leaf=20,
        random_state=42,
    )
else:
    best_name = "RandomForestRegressor"
    final_validation_score = rmse_rf
    best_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

print(f"Model selected: {best_name}")
print(f"Final Validation Performance: {final_validation_score}")

# Fit final models on full training data and ensemble predictions
full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

full_vectorizer = skrub.TableVectorizer()
X_full_vec = X_full.skb.apply(full_vectorizer)

full_pred_gbr = X_full_vec.skb.apply(
    HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=7,
        max_iter=400,
        min_samples_leaf=20,
        random_state=42,
    ),
    y=y_full,
)
full_pred_rf = X_full_vec.skb.apply(
    RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
    y=y_full,
)

full_learner_gbr = full_pred_gbr.skb.make_learner(fitted=True)
full_learner_rf = full_pred_rf.skb.make_learner(fitted=True)

test_env = {"data": test_df.copy()}
test_preds_gbr = full_learner_gbr.predict(test_env)
test_preds_rf = full_learner_rf.predict(test_env)

# Simple ensemble of the two DataOps-trained models
test_preds = 0.6 * np.asarray(test_preds_gbr) + 0.4 * np.asarray(test_preds_rf)

submission = pd.DataFrame({"median_house_value": test_preds})
submission.to_csv("submission.csv", index=False)
