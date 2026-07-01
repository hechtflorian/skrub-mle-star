
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

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

# Small feature views for diversity
reduced_drop_cols = [
    "rooms_per_person",
    "bedrooms_per_household",
    "log_total_rooms",
    "log_population",
    "log_households",
]
X_full_reduced = X_full.drop(columns=[c for c in reduced_drop_cols if c in X_full.columns])
test_df_reduced = test_df.drop(columns=[c for c in reduced_drop_cols if c in test_df.columns])

members = []
val_preds_list = []

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def add_member(name, val_preds, test_preds):
    score = rmse(y_val, val_preds)
    members.append(
        {
            "name": name,
            "score": score,
            "test_preds": np.asarray(test_preds, dtype=float),
        }
    )
    val_preds_list.append(np.asarray(val_preds, dtype=float))
    print(f"Ablation[{name}] RMSE: {score}")

if skrub is not None:
    # DataOps primary branch: original feature set
    data = skrub.var("data", train_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer_1 = skrub.TableVectorizer()
    model_1 = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=7,
        max_iter=400,
        min_samples_leaf=20,
        random_state=42,
    )
    pred_1 = X.skb.apply(vectorizer_1).skb.apply(model_1, y=y)
    learner_1 = pred_1.skb.make_learner(fitted=True)
    val_preds_1 = learner_1.predict({"data": train_df.iloc[va_idx].copy()})
    full_learner_1 = pred_1.skb.make_learner(fitted=True)
    test_preds_1 = full_learner_1.predict({"data": test_df.copy()})
    add_member("skrub_full", val_preds_1, test_preds_1)

    # DataOps secondary branch: reduced feature view
    data_red = skrub.var("data_red", pd.concat([X_full_reduced, train_df[target_col]], axis=1))
    X_red = data_red.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_red = data_red[target_col].skb.mark_as_y()

    vectorizer_2 = skrub.TableVectorizer()
    model_2 = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=25,
        random_state=42,
    )
    pred_2 = X_red.skb.apply(vectorizer_2).skb.apply(model_2, y=y_red)
    learner_2 = pred_2.skb.make_learner(fitted=True)
    val_red_env = pd.concat([X_full_reduced.iloc[va_idx].copy(), train_df[[target_col]].iloc[va_idx].copy()], axis=1)
    val_preds_2 = learner_2.predict({"data_red": val_red_env})
    full_learner_2 = pred_2.skb.make_learner(fitted=True)
    test_red_env = pd.concat([test_df_reduced.copy(), pd.DataFrame({target_col: np.zeros(len(test_df_reduced))})], axis=1)
    test_preds_2 = full_learner_2.predict({"data_red": test_red_env})
    add_member("skrub_reduced", val_preds_2, test_preds_2)

else:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_cols_full = X_full.columns.tolist()
    numeric_cols_reduced = X_full_reduced.columns.tolist()

    sklearn_model_1 = Pipeline(
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
    sklearn_model_1.fit(X_train[numeric_cols_full], y_train)
    val_preds_1 = sklearn_model_1.predict(X_val[numeric_cols_full])
    sklearn_model_1.fit(X_full[numeric_cols_full], y_full)
    test_preds_1 = sklearn_model_1.predict(test_df[numeric_cols_full])
    add_member("sklearn_full", val_preds_1, test_preds_1)

    sklearn_model_2 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=350,
                    min_samples_leaf=25,
                    random_state=42,
                ),
            ),
        ]
    )
    sklearn_model_2.fit(X_train[numeric_cols_reduced], y_train)
    val_preds_2 = sklearn_model_2.predict(X_val[numeric_cols_reduced])
    sklearn_model_2.fit(X_full_reduced[numeric_cols_reduced], y_full)
    test_preds_2 = sklearn_model_2.predict(test_df_reduced[numeric_cols_reduced])
    add_member("sklearn_reduced", val_preds_2, test_preds_2)

# Ensemble predictions
scores = np.array([m["score"] for m in members], dtype=float)
pred_matrix = np.vstack([m["test_preds"] for m in members])

if len(members) == 1:
    weights = np.array([1.0], dtype=float)
    final_test_preds = pred_matrix[0]
    final_validation_score = float(scores[0])
else:
    if np.max(scores) - np.min(scores) < 1e-3:
        weights = np.ones(len(scores), dtype=float) / len(scores)
    else:
        inv = 1.0 / np.maximum(scores, 1e-9)
        weights = inv / inv.sum()
    final_test_preds = np.average(pred_matrix, axis=0, weights=weights)
    final_validation_score = float(np.average(scores, weights=weights))

print(f"Final Validation Performance: {final_validation_score}")

# Write submission
submission = pd.DataFrame({"median_house_value": final_test_preds})
submission.to_csv("submission.csv", index=False)
