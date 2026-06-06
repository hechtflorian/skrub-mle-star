
import os
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

try:
    import skrub
except ImportError:
    skrub = None

# Load data
train_path = "./input/train.csv"
test_path = "./input/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

# Basic feature engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + 1e-6)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + 1e-6)
    df["population_per_household"] = df["population"] / (df["households"] + 1e-6)
    df["rooms_per_person"] = df["total_rooms"] / (df["population"] + 1e-6)
    df["bedrooms_per_household"] = df["total_bedrooms"] / (df["households"] + 1e-6)
    df["income_x_age"] = df["median_income"] * df["housing_median_age"]
    df["log_total_rooms"] = np.log1p(df["total_rooms"])
    df["log_population"] = np.log1p(df["population"])
    df["log_households"] = np.log1p(df["households"])
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

X_full = train_df.drop(columns=[target_col])
y_full = train_df[target_col].values

# Train/validation split
rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(len(idx) * 0.9)
tr_idx, va_idx = idx[:split], idx[split:]

X_train, X_val = X_full.iloc[tr_idx].copy(), X_full.iloc[va_idx].copy()
y_train, y_val = y_full[tr_idx], y_full[va_idx]

# Use skrub DataOps if available, otherwise fall back cleanly to sklearn
if skrub is not None:
    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Keep the DataOps structure intact
    vectorizer = skrub.TableVectorizer()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=7,
        max_iter=400,
        min_samples_leaf=20,
        random_state=42,
    )

    pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
    learner = pred.skb.make_learner(fitted=True)
    val_preds = learner.predict({"data": train_df.iloc[va_idx].copy()})
    final_validation_score = mean_squared_error(y_val, val_preds) ** 0.5

    # Fit on full data for test prediction
    full_learner = pred.skb.make_learner(fitted=True)
    full_learner = pred.skb.make_learner(fitted=True)
    test_preds = full_learner.predict({"data": test_df.copy()})
else:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_cols = X_full.columns.tolist()
    sklearn_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=7,
                    max_iter=400,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )
    sklearn_model.fit(X_train[numeric_cols], y_train)
    val_preds = sklearn_model.predict(X_val[numeric_cols])
    final_validation_score = mean_squared_error(y_val, val_preds) ** 0.5
    sklearn_model.fit(X_full[numeric_cols], y_full)
    test_preds = sklearn_model.predict(test_df[numeric_cols])

print(f"Final Validation Performance: {final_validation_score}")

# Write submission
submission = pd.DataFrame({"median_house_value": test_preds})
submission.to_csv("submission.csv", index=False)
