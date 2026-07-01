

import os
import pandas as pd
import numpy as np
import skrub
from skrub import Cleaner, DropCols, ApplyToCols, SquashingScaler
from skrub import selectors as s
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target = "median_house_value"

data = skrub.var("data", train_df)

# Compact derived-features block: preserve informative geo ratio structures
# and add a few stable density-style ratios while staying DataOps-native.
@skrub.deferred
def add_geo_ratio_features(df):
    out = df.copy()

    def safe_div(num, den):
        den = den.replace(0, np.nan)
        return (num / den).replace([np.inf, -np.inf], np.nan)

    if "total_rooms" in out.columns and "households" in out.columns:
        out["rooms_per_household"] = safe_div(out["total_rooms"], out["households"]).fillna(0.0)
    if "total_bedrooms" in out.columns and "households" in out.columns:
        out["bedrooms_per_household"] = safe_div(out["total_bedrooms"], out["households"]).fillna(0.0)
    if "population" in out.columns and "households" in out.columns:
        out["population_per_household"] = safe_div(out["population"], out["households"]).fillna(0.0)
    if "population" in out.columns and "total_rooms" in out.columns:
        out["population_per_room"] = safe_div(out["population"], out["total_rooms"]).fillna(0.0)
    if "total_bedrooms" in out.columns and "total_rooms" in out.columns:
        out["bedroom_ratio"] = safe_div(out["total_bedrooms"], out["total_rooms"]).fillna(0.0)
    if "latitude" in out.columns and "longitude" in out.columns:
        out["geo_l1"] = (out["latitude"].abs() + out["longitude"].abs()).fillna(0.0)

    return out

data = data.skb.apply_func(add_geo_ratio_features)

X = data.drop(columns=target, errors="ignore").skb.mark_as_X()
y = data[target].skb.mark_as_y()

# Cleaning / redundancy routing
X_clean = X.skb.apply(Cleaner(drop_if_constant=True))

# Route null-heavy columns separately for robust handling.
null_heavy = s.has_nulls()
numeric_cols = s.numeric()
categorical_cols = (s.string() | s.categorical()) - null_heavy

# Numeric path: light squashing for outlier robustness, then median imputation.
X_num = X_clean.skb.select(numeric_cols).skb.apply(
    ApplyToCols(SquashingScaler(max_absolute_value=3), cols=s.all())
)
X_num = X_num.skb.apply(SimpleImputer(strategy="median"))

# Categorical/string path: separate imputation + ordinal encoding to avoid
# a one-size-fits-all vectorizer on mixed feature types.
X_cat = X_clean.skb.select(categorical_cols).skb.apply(
    ApplyToCols(SimpleImputer(strategy="most_frequent"), cols=s.all())
)
X_cat = X_cat.skb.apply(
    ApplyToCols(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
        cols=s.all(),
    )
)

# Keep any remaining columns without forcing them through the categorical encoder.
remainder_cols = s.all() - numeric_cols - categorical_cols
X_rest = X_clean.skb.select(remainder_cols)

# If any remainder exists, vectorize it conservatively; otherwise keep empty path.
if len(remainder_cols.expand(train_df)) > 0:
    X_rest = X_rest.skb.apply(skrub.TableVectorizer())
else:
    X_rest = X_rest

# Concatenate the routed feature blocks.
X_proc = X_num.skb.concat([X_cat, X_rest], axis=1)

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

predictor = X_proc.skb.apply(model, y=y)

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
