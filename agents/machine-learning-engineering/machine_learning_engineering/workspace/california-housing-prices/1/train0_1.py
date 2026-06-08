
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "median_house_value"
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
RANDOM_STATE = 42


def fill_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else ""
            df[col] = df[col].fillna(fill_value)
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9
    if "total_rooms" in df.columns and "households" in df.columns:
        df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    if "total_bedrooms" in df.columns and "total_rooms" in df.columns:
        df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    if "population" in df.columns and "households" in df.columns:
        df["population_per_household"] = df["population"] / (df["households"] + eps)
    return df


def build_sklearn_pipeline(X_train: pd.DataFrame, model):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    return pipe


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

data = skrub.var("data", train_df)
X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

X_clean = X.skb.apply_func(fill_missing_numeric)
X_feat = X_clean.skb.apply_func(add_ratio_features)

vectorizer = skrub.TableVectorizer()
X_vec = X_feat.skb.apply(vectorizer)

hgb_model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=300,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=RANDOM_STATE,
)

hgb_pred = X_vec.skb.apply(hgb_model, y=y)
hgb_learner = hgb_pred.skb.make_learner(fitted=True)

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    min_samples_leaf=2,
)

rf_pred = X_vec.skb.apply(rf_model, y=y)
rf_learner = rf_pred.skb.make_learner(fitted=True)

train_part, valid_part = train_test_split(
    train_df, test_size=0.2, random_state=RANDOM_STATE
)

train_data = skrub.var("data", train_part)
X_train = train_data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_train = train_data[TARGET].skb.mark_as_y()

X_train_clean = X_train.skb.apply_func(fill_missing_numeric)
X_train_feat = X_train_clean.skb.apply_func(add_ratio_features)
X_train_vec = X_train_feat.skb.apply(vectorizer)

hgb_train_pred = X_train_vec.skb.apply(hgb_model, y=y_train)
hgb_train_learner = hgb_train_pred.skb.make_learner(fitted=True)

rf_train_pred = X_train_vec.skb.apply(rf_model, y=y_train)
rf_train_learner = rf_train_pred.skb.make_learner(fitted=True)

hgb_val_pred = hgb_train_learner.predict({"data": valid_part})
rf_val_pred = rf_train_learner.predict({"data": valid_part})

ensemble_val_pred = 0.5 * hgb_val_pred + 0.5 * rf_val_pred
final_validation_score = mean_squared_error(valid_part[TARGET], ensemble_val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

X_tr = train_part.drop(columns=[TARGET]).copy()
y_tr = train_part[TARGET].values
X_va = valid_part.drop(columns=[TARGET]).copy()
y_va = valid_part[TARGET].values

X_tr = fill_missing_numeric(X_tr)
X_va = fill_missing_numeric(X_va)
X_tr = add_ratio_features(X_tr)
X_va = add_ratio_features(X_va)

hgb_pipe = build_sklearn_pipeline(
    X_tr,
    HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    ),
)
hgb_pipe.fit(X_tr, y_tr)
hgb_sklearn_val = hgb_pipe.predict(X_va)

rf_pipe = build_sklearn_pipeline(
    X_tr,
    RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
)
rf_pipe.fit(X_tr, y_tr)
rf_sklearn_val = rf_pipe.predict(X_va)

ensemble_val_pred_2 = 0.5 * hgb_sklearn_val + 0.5 * rf_sklearn_val
_ = mean_squared_error(y_va, ensemble_val_pred_2) ** 0.5

X_full = train_df.drop(columns=[TARGET]).copy()
y_full = train_df[TARGET].values
X_full = fill_missing_numeric(X_full)
X_full = add_ratio_features(X_full)
test_clean = fill_missing_numeric(test_df.copy())
test_clean = add_ratio_features(test_clean)

hgb_final_pipe = build_sklearn_pipeline(
    X_full,
    HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    ),
)
hgb_final_pipe.fit(X_full, y_full)
hgb_test_pred = hgb_final_pipe.predict(test_clean)

rf_final_pipe = build_sklearn_pipeline(
    X_full,
    RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
)
rf_final_pipe.fit(X_full, y_full)
rf_test_pred = rf_final_pipe.predict(test_clean)

test_pred = 0.5 * hgb_test_pred + 0.5 * rf_test_pred

submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
