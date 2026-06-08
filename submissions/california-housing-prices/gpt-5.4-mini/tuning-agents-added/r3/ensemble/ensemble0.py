
import os
import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target = "median_house_value"

# Use the same fixed split for both models
train_idx = np.random.RandomState(42).permutation(len(train_df))
val_size = int(len(train_df) * 0.2)
val_idx = train_idx[:val_size]
fit_idx = train_idx[val_size:]

fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

# -------------------------
# Solution 1 pipeline
# -------------------------
data_1 = skrub.var("data", train_df)

@skrub.deferred
def add_geo_features(df):
    out = df.copy()
    if {"latitude", "longitude"}.issubset(out.columns):
        out["lat_abs"] = out["latitude"].abs()
        out["lon_abs"] = out["longitude"].abs()
        out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
        out["geo_radius"] = np.sqrt(out["latitude"] ** 2 + out["longitude"] ** 2)
    if {"total_rooms", "total_bedrooms", "population", "households"}.issubset(out.columns):
        households = out["households"].replace(0, np.nan)
        total_rooms = out["total_rooms"].replace(0, np.nan)
        total_bedrooms = out["total_bedrooms"].replace(0, np.nan)
        population = out["population"].replace(0, np.nan)

        out["rooms_per_household"] = (out["total_rooms"] / households).replace(
            [np.inf, -np.inf], np.nan
        )
        out["bedrooms_per_room"] = (out["total_bedrooms"] / total_rooms).replace(
            [np.inf, -np.inf], np.nan
        )
        out["population_per_household"] = (out["population"] / households).replace(
            [np.inf, -np.inf], np.nan
        )
        out["bedrooms_per_household"] = (out["total_bedrooms"] / households).replace(
            [np.inf, -np.inf], np.nan
        )

        out["rooms_per_household"] = out["rooms_per_household"].fillna(0.0)
        out["bedrooms_per_room"] = out["bedrooms_per_room"].fillna(0.0)
        out["population_per_household"] = out["population_per_household"].fillna(0.0)
        out["bedrooms_per_household"] = out["bedrooms_per_household"].fillna(0.0)
    return out

data_fe_1 = data_1.skb.apply_func(add_geo_features)

X_1 = data_fe_1.drop(columns=target, errors="ignore").skb.mark_as_X()
y_1 = data_fe_1[target].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.StringEncoder(),
)

X_vec_1 = X_1.skb.apply(vectorizer_1)

model_1 = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

predictor_1 = X_vec_1.skb.apply(model_1, y=y_1)

fitted_learner_1 = predictor_1.skb.make_learner(fitted=True)
_ = fitted_learner_1.predict({"data": fit_df})
val_pred_1 = fitted_learner_1.predict({"data": val_df})
_ = fitted_learner_1.predict({"data": train_df})
test_pred_1 = fitted_learner_1.predict({"data": test_df})

rmse_1 = mean_squared_error(val_df[target].values, val_pred_1) ** 0.5
print(f"Solution 1 Validation RMSE: {rmse_1}")

# -------------------------
# Solution 2 pipeline
# -------------------------
data_2 = skrub.var("data", train_df)

@skrub.deferred
def add_geo_features_v2(df):
    out = df.copy()
    if {"latitude", "longitude"}.issubset(out.columns):
        out["lat_abs"] = out["latitude"].abs()
        out["lon_abs"] = out["longitude"].abs()
        out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
        out["geo_radius"] = np.sqrt(out["latitude"] ** 2 + out["longitude"] ** 2)
    if {"total_rooms", "total_bedrooms", "population", "households"}.issubset(out.columns):
        households = out["households"].replace(0, np.nan)
        total_rooms = out["total_rooms"].replace(0, np.nan)
        total_bedrooms = out["total_bedrooms"].replace(0, np.nan)
        population = out["population"].replace(0, np.nan)

        out["rooms_per_household"] = (out["total_rooms"] / households).replace(
            [np.inf, -np.inf], np.nan
        )
        out["bedrooms_per_room"] = (out["total_bedrooms"] / total_rooms).replace(
            [np.inf, -np.inf], np.nan
        )
        out["population_per_household"] = (out["population"] / households).replace(
            [np.inf, -np.inf], np.nan
        )
        out["bedrooms_per_household"] = (out["total_bedrooms"] / households).replace(
            [np.inf, -np.inf], np.nan
        )

        out["rooms_per_household"] = out["rooms_per_household"].fillna(0.0)
        out["bedrooms_per_room"] = out["bedrooms_per_room"].fillna(0.0)
        out["population_per_household"] = out["population_per_household"].fillna(0.0)
        out["bedrooms_per_household"] = out["bedrooms_per_household"].fillna(0.0)
    return out

data_fe_2 = data_2.skb.apply_func(add_geo_features_v2)

X_2 = data_fe_2.drop(columns=target, errors="ignore").skb.mark_as_X()
y_2 = data_fe_2[target].skb.mark_as_y()

vectorizer_2 = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.StringEncoder(),
)

X_vec_2 = X_2.skb.apply(vectorizer_2)

model_2 = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

predictor_2 = X_vec_2.skb.apply(model_2, y=y_2)

fitted_learner_2 = predictor_2.skb.make_learner(fitted=True)
_ = fitted_learner_2.predict({"data": fit_df})
val_pred_2 = fitted_learner_2.predict({"data": val_df})
_ = fitted_learner_2.predict({"data": train_df})
test_pred_2 = fitted_learner_2.predict({"data": test_df})

rmse_2 = mean_squared_error(val_df[target].values, val_pred_2) ** 0.5
print(f"Solution 2 Validation RMSE: {rmse_2}")

# -------------------------
# Ensemble merge layer
# -------------------------
eps = 1e-12
w1 = 1.0 / (rmse_1 + eps)
w2 = 1.0 / (rmse_2 + eps)
w_sum = w1 + w2
w1 /= w_sum
w2 /= w_sum

ens_val = w1 * np.asarray(val_pred_1) + w2 * np.asarray(val_pred_2)
ens_test = w1 * np.asarray(test_pred_1) + w2 * np.asarray(test_pred_2)

final_validation_score = mean_squared_error(val_df[target].values, ens_val) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({"median_house_value": ens_test})
submission.to_csv("submission.csv", index=False)
