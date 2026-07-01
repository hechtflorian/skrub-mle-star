import os
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

train_path = os.path.join("./input", "train.csv")
train = pd.read_csv(train_path)

target_col = "median_house_value"

# Train/validation split: last 20% as validation, matching the original script
n = len(train)
val_size = max(1, int(0.2 * n))
train_data = train.iloc[:-val_size].reset_index(drop=True)
val_data = train.iloc[-val_size:].reset_index(drop=True)

def fit_and_score(df_train, df_val, use_vectorizer=True, model_params=None, variant_name="baseline"):
    model_params = model_params or {}

    data = skrub.var("data", df_train)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    estimator = HistGradientBoostingRegressor(random_state=42, **model_params)

    if use_vectorizer:
        vectorizer = skrub.TableVectorizer(
            high_cardinality=skrub.choose_from(
                {
                    "minhash": skrub.MinHashEncoder(n_components=16),
                    "gap": skrub.GapEncoder(n_components=16),
                },
                name="high_card_encoder",
            )
        )
        pred_graph = X.skb.apply(vectorizer).skb.apply(estimator, y=y)
    else:
        # Ablation: remove TableVectorizer, use raw features where possible
        pred_graph = X.skb.apply(estimator, y=y)

    learner = pred_graph.skb.make_learner(fitted=True)
    val_pred = learner.predict({"data": df_val})
    rmse = mean_squared_error(df_val[target_col], val_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse:.6f}")
    return rmse

results = {}

# Baseline: original-style pipeline
results["baseline"] = fit_and_score(
    train_data,
    val_data,
    use_vectorizer=True,
    model_params=dict(
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=25,
    ),
    variant_name="baseline",
)

# Ablation 1: remove the vectorizer
results["no_vectorizer"] = fit_and_score(
    train_data,
    val_data,
    use_vectorizer=False,
    model_params=dict(
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=25,
    ),
    variant_name="no_vectorizer",
)

# Ablation 2: simplify the tree model
results["shallower_model"] = fit_and_score(
    train_data,
    val_data,
    use_vectorizer=True,
    model_params=dict(
        learning_rate=0.06,
        max_depth=3,
        min_samples_leaf=25,
    ),
    variant_name="shallower_model",
)

baseline_score = results["baseline"]
deltas = {k: v - baseline_score for k, v in results.items() if k != "baseline"}
worst_variant = max(deltas, key=lambda k: deltas[k])  # largest RMSE increase
best_contributor = worst_variant
print(f"Baseline RMSE: {baseline_score:.6f}")
for name, delta in deltas.items():
    sign = "+" if delta >= 0 else ""
    print(f"Change vs baseline for {name}: {sign}{delta:.6f} RMSE")
print(f"Most important part of the pipeline: {best_contributor}")