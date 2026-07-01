

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "median_house_value"

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

@skrub.deferred
def add_derived_features(df):
    out = df.copy()
    households = out["households"].replace(0, np.nan)
    out["rooms_per_household"] = (out["total_rooms"] / households).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["population_per_household"] = (out["population"] / households).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

data = skrub.var("data", train_part)
data_fe = data.skb.apply_func(add_derived_features)

X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()

# Explicit routing:
# - numeric columns are passed through as-is
# - string / categorical columns get a light encoder strategy
numeric_cols = train_part.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_part.drop(columns=[target_col], errors="ignore").select_dtypes(exclude=[np.number]).columns.tolist()

if len(numeric_cols) > 0:
    X_num = X.skb.select(numeric_cols)
    X_num_enc = X_num.skb.apply(
        skrub.ApplyToCols(
            skrub.TableVectorizer(
                low_cardinality=skrub.ToCategorical(),
                high_cardinality=skrub.StringEncoder(),
            ),
            cols=numeric_cols,
        )
    )
else:
    X_num_enc = None

if len(categorical_cols) > 0:
    X_cat = X.skb.select(categorical_cols)
    X_cat_enc = X_cat.skb.apply(
        skrub.ApplyToCols(
            skrub.TableVectorizer(
                low_cardinality=skrub.ToCategorical(),
                high_cardinality=skrub.StringEncoder(),
            ),
            cols=categorical_cols,
        )
    )
else:
    X_cat_enc = None

if X_num_enc is not None and X_cat_enc is not None:
    X_vec = X_num_enc.skb.concat([X_cat_enc], axis=1)
elif X_num_enc is not None:
    X_vec = X_num_enc
elif X_cat_enc is not None:
    X_vec = X_cat_enc
else:
    X_vec = X

model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=4000,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

pred = X_vec.skb.apply(model, y=y)

learner = pred.skb.make_learner(fitted=True)
learner.fit({"data": train_part})

valid_pred = learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

test_pred = learner.predict({"data": test_df})
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
