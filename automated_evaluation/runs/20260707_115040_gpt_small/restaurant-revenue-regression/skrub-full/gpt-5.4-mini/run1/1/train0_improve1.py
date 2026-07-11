
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

random_state = 42
target_col = "revenue"

train_path = glob.glob(os.path.join("./input", "train.csv"))[0]
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()


import numpy as np
import pandas as pd
import skrub
import skrub.selectors as s
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)

def add_date_features(df):
    out = df.copy()
    if "Open Date" in out.columns:
        dt = pd.to_datetime(out["Open Date"], errors="coerce")
        out["Open Date_year"] = dt.dt.year
        out["Open Date_month"] = dt.dt.month
        out["Open Date_dayofweek"] = dt.dt.dayofweek
        out["Open Date_dayofyear"] = dt.dt.dayofyear
        out["Open Date_age_days"] = (pd.Timestamp("today").normalize() - dt).dt.days
    return out

data_fe = data_train.skb.apply_func(add_date_features)

X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()

present_cat_cols = [c for c in ["City", "City Group", "Type"] if c in train_part.columns]

X_no_id = X.skb.apply(skrub.DropCols(cols=s.cols("Id")))
X_main = X_no_id.skb.apply(skrub.DropCols(cols=s.cols("Open Date")))

if present_cat_cols:
    X_rest = X_main.skb.apply(skrub.DropCols(cols=s.cols(*present_cat_cols)))
else:
    X_rest = X_main

vectorizer = skrub.TableVectorizer(
    low_cardinality="drop",
    high_cardinality="drop",
)

rest_vec = X_rest.skb.apply(vectorizer)

if present_cat_cols:
    cat_path = X_main.skb.select(s.cols(*present_cat_cols)).skb.apply(
        ColumnTransformer(
            transformers=[
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    slice(None),
                )
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    )
    X_final = rest_vec.skb.concat([cat_path], axis=1)
else:
    X_final = rest_vec

model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=random_state,
    verbose=0,
)

pred = X_final.skb.apply(model, y=y)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

