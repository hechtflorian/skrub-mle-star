import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def evaluate_variant(variant_name, train_frame, vectorizer, model):
    data_train = skrub.var("data", train_frame)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    val_learner = pred.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})
    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return rmse

base_vectorizer = skrub.TableVectorizer()
base_model = CatBoostRegressor(
    iterations=5000,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200,
    allow_writing_files=False,
)

results = {}

# 1) Baseline
results["baseline"] = evaluate_variant(
    "baseline",
    train_part,
    base_vectorizer,
    base_model,
)

# 2) Redundancy ablation: drop one strongly correlated feature (households)
train_drop_households = train_part.drop(columns=["households"], errors="ignore").copy()
results["drop_households"] = evaluate_variant(
    "drop_households",
    train_drop_households,
    skrub.TableVectorizer(),
    CatBoostRegressor(
        iterations=5000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200,
        allow_writing_files=False,
    ),
)

# 3) Structural feature ablation: add a ratio feature (rooms per household)
@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    if "total_rooms" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (
            out["total_rooms"] / denom
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "population" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["people_per_household"] = (
            out["population"] / denom
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

data_train_fe = skrub.var("data", train_part)
data_train_fe = data_train_fe.skb.apply_func(add_ratio_features)
X_train_fe = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_fe = data_train_fe[target_col].skb.mark_as_y()

pred_fe = X_train_fe.skb.apply(skrub.TableVectorizer()).skb.apply(
    CatBoostRegressor(
        iterations=5000,
        depth=8,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200,
        allow_writing_files=False,
    ),
    y=y_train_fe,
)
val_learner_fe = pred_fe.skb.make_learner(fitted=True)
valid_pred_fe = val_learner_fe.predict({"data": valid_part})
results["add_ratios"] = mean_squared_error(valid_part[target_col], valid_pred_fe) ** 0.5
print(f"Ablation[add_ratios] RMSE: {results['add_ratios']}")

# 4) Model ablation: shallower tree
results["shallower_depth6"] = evaluate_variant(
    "shallower_depth6",
    train_part,
    skrub.TableVectorizer(),
    CatBoostRegressor(
        iterations=5000,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200,
        allow_writing_files=False,
    ),
)

best_variant = min(results, key=results.get)
best_score = results[best_variant]
baseline_score = results["baseline"]

print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")
print(
    f"Most important change vs baseline: "
    f"{best_variant} improved by {baseline_score - best_score:.6f} RMSE"
)
print(f"Final Validation Performance: {best_score}")