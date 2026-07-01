
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")
final_dir = "./final"
os.makedirs(final_dir, exist_ok=True)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target = "median_house_value"

data = skrub.var("data", train_df)

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

data_fe = data.skb.apply_func(add_geo_features)

X = data_fe.drop(columns=target, errors="ignore").skb.mark_as_X()
y = data_fe[target].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.StringEncoder(),
)

X_vec = X.skb.apply(vectorizer)

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

model_2 = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=7,
)

predictor_1 = X_vec.skb.apply(model_1, y=y)
predictor_2 = X_vec.skb.apply(model_2, y=y)

train_idx = np.random.RandomState(42).permutation(len(train_df))
val_size = int(len(train_df) * 0.2)
val_idx = train_idx[:val_size]
fit_idx = train_idx[val_size:]

fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

fitted_learner_1 = predictor_1.skb.make_learner(fitted=True)
fitted_learner_2 = predictor_2.skb.make_learner(fitted=True)

_ = fitted_learner_1.predict({"data": fit_df})
_ = fitted_learner_2.predict({"data": fit_df})

val_pred_1 = fitted_learner_1.predict({"data": val_df})
val_pred_2 = fitted_learner_2.predict({"data": val_df})

y_true = val_df[target].values

final_learner_1 = predictor_1.skb.make_learner(fitted=True)
final_learner_2 = predictor_2.skb.make_learner(fitted=True)

_ = final_learner_1.predict({"data": train_df})
_ = final_learner_2.predict({"data": train_df})

test_pred_1 = final_learner_1.predict({"data": test_df})
test_pred_2 = final_learner_2.predict({"data": test_df})

# --- tiny blend split on validation fold ---
rng = np.random.RandomState(42)
blend_perm = rng.permutation(len(val_df))
blend_size = max(1, int(len(val_df) * 0.5))
blend_fit_idx = blend_perm[:blend_size]
blend_sanity_idx = blend_perm[blend_size:]

v1_fit = val_pred_1[blend_fit_idx]
v2_fit = val_pred_2[blend_fit_idx]
y_fit = y_true[blend_fit_idx]

v1_sanity = val_pred_1[blend_sanity_idx] if len(blend_sanity_idx) > 0 else val_pred_1[blend_fit_idx]
v2_sanity = val_pred_2[blend_sanity_idx] if len(blend_sanity_idx) > 0 else val_pred_2[blend_fit_idx]
y_sanity = y_true[blend_sanity_idx] if len(blend_sanity_idx) > 0 else y_true[blend_fit_idx]

mu = np.array([v1_fit.mean(), v2_fit.mean()])
sigma = np.array([v1_fit.std(), v2_fit.std()])
sigma = np.where(sigma < 1e-8, 1.0, sigma)

Z_fit = np.column_stack([(v1_fit - mu[0]) / sigma[0], (v2_fit - mu[1]) / sigma[1]])
Z_sanity = np.column_stack([(v1_sanity - mu[0]) / sigma[0], (v2_sanity - mu[1]) / sigma[1]])
Z_test = np.column_stack([(test_pred_1 - mu[0]) / sigma[0], (test_pred_2 - mu[1]) / sigma[1]])

ridge = Ridge(alpha=1.0, fit_intercept=True, positive=True, random_state=42)
ridge.fit(Z_fit, y_fit)
coef = np.maximum(ridge.coef_.astype(float), 0.0)
coef = coef / coef.sum()
blend_sanity_pred = ridge.predict(Z_sanity)
sanity_rmse = mean_squared_error(y_sanity, blend_sanity_pred) ** 0.5
ens_test = Z_test @ coef + ridge.intercept_

rmse_1 = mean_squared_error(y_true, val_pred_1) ** 0.5
rmse_2 = mean_squared_error(y_true, val_pred_2) ** 0.5

if rmse_2 < rmse_1:
    base_val = val_pred_2
    base_test = test_pred_2
    corr_feat_val = val_pred_1 - val_pred_2
    corr_feat_test = test_pred_1 - test_pred_2
else:
    base_val = val_pred_1
    base_test = test_pred_1
    corr_feat_val = val_pred_2 - val_pred_1
    corr_feat_test = test_pred_2 - test_pred_1

residual = y_true - base_val
corr_scale = 0.25
corr_model = Ridge(alpha=1.0, fit_intercept=True)
corr_model.fit(corr_feat_val.reshape(-1, 1), residual)
corr_pred_test = corr_model.predict(corr_feat_test.reshape(-1, 1))
residual_blend_test = base_test + corr_scale * corr_pred_test

test_pred = 0.7 * residual_blend_test + 0.3 * ens_test

if rmse_2 < rmse_1:
    base_val_full = val_pred_2
    corr_feat_val_full = val_pred_1 - val_pred_2
else:
    base_val_full = val_pred_1
    corr_feat_val_full = val_pred_2 - val_pred_1

corr_pred_val_full = corr_model.predict(corr_feat_val_full.reshape(-1, 1))
val_residual_blend = base_val_full + corr_scale * corr_pred_val_full

final_validation_score = mean_squared_error(y_true, val_residual_blend) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Refit on full training set for final submission
full_learner_1 = predictor_1.skb.make_learner(fitted=True)
full_learner_2 = predictor_2.skb.make_learner(fitted=True)
full_pred_1 = full_learner_1.predict({"data": train_df})
full_pred_2 = full_learner_2.predict({"data": train_df})

full_test_pred_1 = full_learner_1.predict({"data": test_df})
full_test_pred_2 = full_learner_2.predict({"data": test_df})

rmse_full_1 = mean_squared_error(train_df[target].values, full_pred_1) ** 0.5
rmse_full_2 = mean_squared_error(train_df[target].values, full_pred_2) ** 0.5

if rmse_full_2 < rmse_full_1:
    base_full_test = full_test_pred_2
    corr_feat_full_train = full_pred_1 - full_pred_2
    corr_feat_full_test = full_test_pred_1 - full_test_pred_2
else:
    base_full_test = full_test_pred_1
    corr_feat_full_train = full_pred_2 - full_pred_1
    corr_feat_full_test = full_test_pred_2 - full_test_pred_1

full_residual = train_df[target].values - np.where(rmse_full_2 < rmse_full_1, full_pred_2, full_pred_1)
full_corr_model = Ridge(alpha=1.0, fit_intercept=True)
full_corr_model.fit(corr_feat_full_train.reshape(-1, 1), full_residual)
full_corr_test = full_corr_model.predict(corr_feat_full_test.reshape(-1, 1))
final_test_pred = base_full_test + corr_scale * full_corr_test

submission = pd.DataFrame({"median_house_value": final_test_pred})
submission.to_csv(os.path.join(final_dir, "submission.csv"), index=False)
