
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

model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=28,
    min_child_samples=55,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.16443457513284063,
    random_state=42,
)

predictor = X_vec.skb.apply(model, y=y)

train_idx = np.random.RandomState(42).permutation(len(train_df))
val_size = int(len(train_df) * 0.2)
val_idx = train_idx[:val_size]
fit_idx = train_idx[val_size:]

fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

fitted_learner = predictor.skb.make_learner(fitted=True)

_ = fitted_learner.predict({"data": fit_df})
val_pred = fitted_learner.predict({"data": val_df})

y_true = val_df[target].values
rmse = mean_squared_error(y_true, val_pred) ** 0.5
print(f"Final Validation Performance: {rmse}")

final_learner = predictor.skb.make_learner(fitted=True)
_ = final_learner.predict({"data": train_df})
test_pred = final_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
