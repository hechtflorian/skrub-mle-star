
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "median_house_value"
eps = 1e-8

def build_member(train_part, valid_part, model_random_seed=42):
    @skrub.deferred
    def add_housing_ratio_features(df):
        out = df.copy()
        if "households" in out.columns and "total_rooms" in out.columns:
            denom = out["households"].replace(0, np.nan)
            out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        if "total_rooms" in out.columns and "total_bedrooms" in out.columns:
            denom = out["total_rooms"].replace(0, np.nan)
            out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        if "households" in out.columns and "population" in out.columns:
            denom = out["households"].replace(0, np.nan)
            out["population_per_household"] = (out["population"] / denom).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        return out

    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(add_housing_ratio_features)
    X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.03,
        iterations=800,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
        random_seed=model_random_seed,
        early_stopping_rounds=50,
    )

    pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5

    # Linear calibration on holdout: y_true ≈ a * pred + b
    a, b = np.polyfit(valid_pred, valid_part[target_col].to_numpy(), 1)
    calibrated_valid_pred = a * valid_pred + b
    calibrated_valid_rmse = mean_squared_error(valid_part[target_col], calibrated_valid_pred) ** 0.5

    return {
        "learner": learner,
        "valid_pred": valid_pred,
        "calibrated_valid_pred": calibrated_valid_pred,
        "rmse": rmse,
        "calibrated_rmse": calibrated_valid_rmse,
        "calibration_a": a,
        "calibration_b": b,
    }

# Member 1: baseline split and model seed
train_idx_1, valid_idx_1 = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part_1 = train_df.iloc[train_idx_1].copy()
valid_part_1 = train_df.iloc[valid_idx_1].copy()

# Member 2: different split seed
train_idx_2, valid_idx_2 = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=7
)
train_part_2 = train_df.iloc[train_idx_2].copy()
valid_part_2 = train_df.iloc[valid_idx_2].copy()

# Member 3: same split as baseline, different CatBoost seed
train_part_3 = train_part_1.copy()
valid_part_3 = valid_part_1.copy()

member1 = build_member(train_part_1, valid_part_1, model_random_seed=42)
member2 = build_member(train_part_2, valid_part_2, model_random_seed=42)
member3 = build_member(train_part_3, valid_part_3, model_random_seed=99)

# Use the baseline holdout split for ensemble scoring, as requested
y_true = valid_part_1[target_col].to_numpy()

# Recompute baseline holdout predictions for member 2 and 3 on member 1's holdout only
# by training their same pipeline on their own training parts, then predicting member1 valid_part.
# This keeps the same exact skrub + CatBoost pipeline per member.
def predict_on_reference_valid(train_part, ref_valid_part, model_random_seed=42):
    @skrub.deferred
    def add_housing_ratio_features(df):
        out = df.copy()
        if "households" in out.columns and "total_rooms" in out.columns:
            denom = out["households"].replace(0, np.nan)
            out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        if "total_rooms" in out.columns and "total_bedrooms" in out.columns:
            denom = out["total_rooms"].replace(0, np.nan)
            out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        if "households" in out.columns and "population" in out.columns:
            denom = out["households"].replace(0, np.nan)
            out["population_per_household"] = (out["population"] / denom).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        return out

    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(add_housing_ratio_features)
    X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.03,
        iterations=800,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
        random_seed=model_random_seed,
        early_stopping_rounds=50,
    )

    pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    return learner.predict({"data": ref_valid_part})

# Member 2 and 3 predictions on baseline holdout for ensemble evaluation
valid_pred_2_on_1 = predict_on_reference_valid(train_part_2, valid_part_1, model_random_seed=42)
valid_pred_3_on_1 = predict_on_reference_valid(train_part_3, valid_part_1, model_random_seed=99)

# Calibrate each member using its own validation fold if available;
# for the ensemble metric, we need predictions on the same fixed holdout.
# Member 1 calibration on holdout 1
a1, b1 = member1["calibration_a"], member1["calibration_b"]
cal_pred_1 = a1 * member1["valid_pred"] + b1
rmse1 = mean_squared_error(y_true, cal_pred_1) ** 0.5

# For members 2 and 3, fit calibration on their own holdouts, then apply the same linear correction
a2, b2 = np.polyfit(member2["valid_pred"], valid_part_2[target_col].to_numpy(), 1)
a3, b3 = np.polyfit(member3["valid_pred"], valid_part_3[target_col].to_numpy(), 1)

cal_pred_2 = a2 * valid_pred_2_on_1 + b2
cal_pred_3 = a3 * valid_pred_3_on_1 + b3

rmse2 = mean_squared_error(y_true, cal_pred_2) ** 0.5
rmse3 = mean_squared_error(y_true, cal_pred_3) ** 0.5

weights = np.array([1.0 / (rmse1 + eps), 1.0 / (rmse2 + eps), 1.0 / (rmse3 + eps)])
weights = weights / weights.sum()

final_valid_pred = weights[0] * cal_pred_1 + weights[1] * cal_pred_2 + weights[2] * cal_pred_3
final_validation_score = mean_squared_error(y_true, final_valid_pred) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")
