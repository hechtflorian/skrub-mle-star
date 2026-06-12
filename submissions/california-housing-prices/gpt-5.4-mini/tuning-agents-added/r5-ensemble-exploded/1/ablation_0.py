import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TARGET = "median_house_value"
RANDOM_STATE = 42

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def fit_and_score(train_data, valid_data, vectorizer, model, variant_name):
    data_train = skrub.var("data", train_data)
    X_train = data_train.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y_train = data_train[TARGET].skb.mark_as_y()

    pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_data})
    rmse = mean_squared_error(valid_data[TARGET], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return rmse


base_vectorizer = skrub.TableVectorizer(
    low_cardinality="passthrough",
    high_cardinality=skrub.StringEncoder(n_components=20),
)

base_model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.03,
    iterations=5000,
    random_seed=RANDOM_STATE,
    verbose=200,
)

results = {}

# Baseline: original pipeline
results["baseline"] = fit_and_score(
    train_part,
    valid_part,
    base_vectorizer,
    base_model,
    "baseline",
)

# Ablation 1: simplified encoding for all columns via default TableVectorizer
# Hypothesis: current custom high_cardinality StringEncoder may be unnecessary on this all-numeric dataset.
simple_vectorizer = skrub.TableVectorizer()
results["simple_encoding"] = fit_and_score(
    train_part,
    valid_part,
    simple_vectorizer,
    base_model,
    "simple_encoding",
)

# Ablation 2: drop one redundant column from a highly correlated pair
# Hypothesis: total_bedrooms is largely redundant with households / total_rooms.
@skrub.deferred
def drop_redundant_column(df):
    out = df.copy()
    if "total_bedrooms" in out.columns:
        out = out.drop(columns=["total_bedrooms"])
    return out


data_train_fe = skrub.var("data", train_part).skb.apply_func(drop_redundant_column)
X_train_fe = data_train_fe.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y_train_fe = data_train_fe[TARGET].skb.mark_as_y()

pred_graph_fe = X_train_fe.skb.apply(base_vectorizer).skb.apply(base_model, y=y_train_fe)
learner_fe = pred_graph_fe.skb.make_learner(fitted=True)
valid_pred_fe = learner_fe.predict({"data": valid_part})
rmse_fe = mean_squared_error(valid_part[TARGET], valid_pred_fe) ** 0.5
results["drop_total_bedrooms"] = rmse_fe
print(f"Ablation[drop_total_bedrooms] RMSE: {rmse_fe}")

best_variant = min(results, key=results.get)
best_score = results[best_variant]
baseline_score = results["baseline"]

print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")
print(f"Baseline RMSE: {baseline_score}")
print(f"RMSE improvement over baseline: {baseline_score - best_score}")

if best_variant == "baseline":
    print("Most important part: the original encoding/model setup is already best; no ablation improved it.")
elif best_variant == "simple_encoding":
    print("Most important part: the custom high-cardinality encoding was not helpful; simpler encoding improved performance.")
elif best_variant == "drop_total_bedrooms":
    print("Most important part: removing the redundant total_bedrooms feature improved performance the most.")