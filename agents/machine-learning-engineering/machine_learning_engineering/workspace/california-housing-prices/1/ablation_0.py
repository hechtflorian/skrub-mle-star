import pandas as pd
import numpy as np
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")

target_col = "median_house_value"
train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

def rmse_score(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def fit_eval_pipeline(train_df_in, valid_df_in, build_X_fn, model_params=None):
    data = skrub.var("data", train_df_in)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    X_mod = build_X_fn(X)

    vectorizer = skrub.TableVectorizer()
    X_vec = X_mod.skb.apply(vectorizer)

    params = dict(
        depth=8,
        learning_rate=0.05,
        iterations=4000,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    )
    if model_params:
        params.update(model_params)

    model = CatBoostRegressor(**params)
    pred = X_vec.skb.apply(model, y=y)

    learner = pred.skb.make_learner(fitted=True)
    learner.fit({"data": train_df_in})
    valid_pred = learner.predict({"data": valid_df_in})
    return rmse_score(valid_df_in[target_col], valid_pred)

# Baseline: original pipeline
def baseline_builder(X):
    return X

# Ablation 1: drop one redundant feature from a highly correlated pair
# total_bedrooms and households are extremely correlated (0.97)
def drop_redundant_builder(X):
    return X.skb.apply(skrub.DropCols(cols=["households"]))

# Ablation 2: add a simple ratio feature that may capture density information
@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    denom = out["households"].replace(0, np.nan)
    out["rooms_per_household"] = (out["total_rooms"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["bedrooms_per_household"] = (out["total_bedrooms"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def ratio_builder(X):
    return X.skb.apply_func(add_ratio_features)

# Ablation 3: slightly simpler model to test importance of model capacity
def shallow_model_builder(X):
    return X

results = {}

results["baseline"] = fit_eval_pipeline(train_part, valid_part, baseline_builder)
results["drop_households"] = fit_eval_pipeline(train_part, valid_part, drop_redundant_builder)
results["add_ratio_features"] = fit_eval_pipeline(train_part, valid_part, ratio_builder)
results["shallower_model"] = fit_eval_pipeline(
    train_part,
    valid_part,
    shallow_model_builder,
    model_params={"depth": 6, "iterations": 2500},
)

for name, score in results.items():
    print(f"Ablation[{name}] RMSE: {score:.6f}")

best_name = min(results, key=results.get)
best_score = results[best_name]
baseline_score = results["baseline"]

print(f"Final Validation Performance: {baseline_score:.6f}")
print(f"Best ablation variant: {best_name} | RMSE: {best_score:.6f}")

delta = {k: v - baseline_score for k, v in results.items() if k != "baseline"}
most_important = min(delta, key=lambda k: abs(delta[k]))
largest_improvement = min(delta, key=delta.get)

print(f"Most impactful change by absolute RMSE shift: {most_important} | Delta: {delta[most_important]:+.6f}")
print(f"Best improvement over baseline: {largest_improvement} | Delta: {delta[largest_improvement]:+.6f}")