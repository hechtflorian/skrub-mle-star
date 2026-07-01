import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Load data
train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

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

numeric_cols = X_full.columns.tolist()

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def evaluate_pipeline(name, use_feature_engineering=True, use_scaler=True, model_params=None):
    if use_feature_engineering:
        Xtr = X_train[numeric_cols]
        Xva = X_val[numeric_cols]
    else:
        # No feature engineering ablation: use only original raw features
        raw_cols = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income"
        ]
        Xtr = X_train[raw_cols]
        Xva = X_val[raw_cols]

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler(with_mean=False)))

    params = dict(
        learning_rate=0.05,
        max_depth=7,
        max_iter=400,
        min_samples_leaf=20,
        random_state=42,
    )
    if model_params:
        params.update(model_params)

    steps.append(("model", HistGradientBoostingRegressor(**params)))
    model = Pipeline(steps=steps)

    model.fit(Xtr, y_train)
    preds = model.predict(Xva)
    score = rmse(y_val, preds)
    print(f"Ablation[{name}] RMSE: {score:.6f}")
    return score

results = {}

# Baseline: full feature engineering + scaler
results["baseline"] = evaluate_pipeline(
    name="baseline",
    use_feature_engineering=True,
    use_scaler=True,
)

# Ablation 1: disable feature engineering
results["no_feature_engineering"] = evaluate_pipeline(
    name="no_feature_engineering",
    use_feature_engineering=False,
    use_scaler=True,
)

# Ablation 2: disable scaling
results["no_scaler"] = evaluate_pipeline(
    name="no_scaler",
    use_feature_engineering=True,
    use_scaler=False,
)

# Ablation 3: simpler model
results["simpler_model"] = evaluate_pipeline(
    name="simpler_model",
    use_feature_engineering=True,
    use_scaler=True,
    model_params={"max_depth": 4, "max_iter": 200, "min_samples_leaf": 40},
)

baseline_score = results["baseline"]
print(f"Final Validation Performance: {baseline_score:.6f}")

# Compare ablations against baseline
deltas = {k: v - baseline_score for k, v in results.items() if k != "baseline"}
for name, delta in deltas.items():
    sign = "+" if delta >= 0 else ""
    print(f"Ablation impact[{name}] vs baseline: {sign}{delta:.6f} RMSE")

# Identify the most important component by performance drop when removed / simplified
most_important = max(deltas.items(), key=lambda kv: kv[1])
print(
    f"Most important component: {most_important[0]} "
    f"(largest RMSE increase of {most_important[1]:.6f} when modified)"
)