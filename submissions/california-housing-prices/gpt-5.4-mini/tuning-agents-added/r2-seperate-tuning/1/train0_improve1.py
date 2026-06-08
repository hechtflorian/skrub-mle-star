

import os
import pandas as pd
import numpy as np
import skrub
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

@skrub.deferred
def add_housing_features(df):
    out = df.copy()

    rooms = out["total_rooms"].astype(float)
    bedrooms = out["total_bedrooms"].astype(float)
    population = out["population"].astype(float)
    households = out["households"].astype(float).replace(0, np.nan)

    out["rooms_per_household"] = (rooms / households).replace([np.inf, -np.inf], np.nan)
    out["bedrooms_per_room"] = (bedrooms / rooms.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["population_per_household"] = (population / households).replace([np.inf, -np.inf], np.nan)
    out["bedrooms_per_household"] = (bedrooms / households).replace([np.inf, -np.inf], np.nan)
    out["rooms_per_person"] = (rooms / population.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    return out

data = skrub.var("data", train_df)
data_fe = data.skb.apply_func(add_housing_features)

X = data_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()

# Fixed structural ablation: drop one redundant correlated count feature after ratios are created
X = X.skb.apply(skrub.DropCols(cols=["households"]))

# Clean heavy-tailed numeric variables before routing/vectorization
X = X.skb.apply(skrub.ApplyToCols(skrub.SquashingScaler(max_absolute_value=3), cols=skrub.selectors.numeric()))

# Route low-cardinality categoricals through a simple categorical path
# and high-cardinality strings through a more suitable encoder.
low_card_categorical = skrub.selectors.categorical() & skrub.selectors.cardinality_below(20)
high_card_string = skrub.selectors.string() & ~skrub.selectors.cardinality_below(20)
remaining = skrub.selectors.all() - low_card_categorical - high_card_string

X_low = X.skb.select(low_card_categorical).skb.apply(skrub.ApplyToCols(skrub.ToCategorical(), cols=skrub.selectors.all(), allow_reject=True))
X_high = X.skb.select(high_card_string).skb.apply(skrub.ApplyToCols(skrub.StringEncoder(), cols=skrub.selectors.all(), allow_reject=True))
X_rem = X.skb.select(remaining)

# Recombine routed branches
X_routed = X_rem.skb.concat([X_low, X_high], axis=1)

vectorizer = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.GapEncoder(),
)

X_vec = X_routed.skb.apply(vectorizer)

model = lgb.LGBMRegressor(
    n_estimators=2500,
    learning_rate=0.025,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.2,
    min_child_samples=25,
    random_state=42,
)

predictor = X_vec.skb.apply(model, y=y)
learner = predictor.skb.make_learner(fitted=True)

# Direct holdout RMSE on a real validation split
train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
train_part = train_df.iloc[train_idx].reset_index(drop=True)
val_part = train_df.iloc[val_idx].reset_index(drop=True)

data_tr = skrub.var("data", train_part)
data_va = skrub.var("data", val_part)

data_tr_fe = data_tr.skb.apply_func(add_housing_features)
data_va_fe = data_va.skb.apply_func(add_housing_features)

X_tr = data_tr_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_tr = data_tr_fe[target_col].skb.mark_as_y()

X_va = data_va_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_va = data_va_fe[target_col].skb.mark_as_y()

X_tr = X_tr.skb.apply(skrub.DropCols(cols=["households"]))
X_va = X_va.skb.apply(skrub.DropCols(cols=["households"]))

X_tr = X_tr.skb.apply(skrub.ApplyToCols(skrub.SquashingScaler(max_absolute_value=3), cols=skrub.selectors.numeric()))
X_va = X_va.skb.apply(skrub.ApplyToCols(skrub.SquashingScaler(max_absolute_value=3), cols=skrub.selectors.numeric()))

low_card_categorical = skrub.selectors.categorical() & skrub.selectors.cardinality_below(20)
high_card_string = skrub.selectors.string() & ~skrub.selectors.cardinality_below(20)
remaining = skrub.selectors.all() - low_card_categorical - high_card_string

X_tr_low = X_tr.skb.select(low_card_categorical).skb.apply(skrub.ApplyToCols(skrub.ToCategorical(), cols=skrub.selectors.all(), allow_reject=True))
X_tr_high = X_tr.skb.select(high_card_string).skb.apply(skrub.ApplyToCols(skrub.StringEncoder(), cols=skrub.selectors.all(), allow_reject=True))
X_tr_rem = X_tr.skb.select(remaining)
X_tr_routed = X_tr_rem.skb.concat([X_tr_low, X_tr_high], axis=1)
X_tr_vec = X_tr_routed.skb.apply(vectorizer)

X_va_low = X_va.skb.select(low_card_categorical).skb.apply(skrub.ApplyToCols(skrub.ToCategorical(), cols=skrub.selectors.all(), allow_reject=True))
X_va_high = X_va.skb.select(high_card_string).skb.apply(skrub.ApplyToCols(skrub.StringEncoder(), cols=skrub.selectors.all(), allow_reject=True))
X_va_rem = X_va.skb.select(remaining)
X_va_routed = X_va_rem.skb.concat([X_va_low, X_va_high], axis=1)
X_va_vec = X_va_routed.skb.apply(vectorizer)

val_predictor = X_tr_vec.skb.apply(model, y=y_tr)
val_learner = val_predictor.skb.make_learner(fitted=True)

val_preds = val_learner.predict({"data": val_part})
rmse = mean_squared_error(val_part[target_col], val_preds) ** 0.5

print(f"Final Validation Performance: {rmse}")

test_preds = learner.predict({"data": test_df})
submission = pd.DataFrame({target_col: test_preds})
submission.to_csv("submission.csv", index=False)
