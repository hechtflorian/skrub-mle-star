import os
import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

train_df = pd.read_csv(TRAIN_PATH)
target_col = "median_house_value"

# Fixed holdout split
rng = np.random.RandomState(0)
indices = np.arange(len(train_df))
rng.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]

train_subset = train_df.iloc[train_idx].copy()
val_subset = train_df.iloc[val_idx].copy()


def run_variant(name, use_vectorizer=True, use_target=True):
    """
    Simple ablations:
    - disable TableVectorizer
    - disable target usage (train an untrained model path is not meaningful, so we
      approximate by removing the supervised signal via a constant baseline)
    """
    if not use_target:
        # Baseline: predict the training mean of the target
        pred = np.full(len(val_subset), train_subset[target_col].mean())
        rmse = mean_squared_error(val_subset[target_col], pred) ** 0.5
        print(f"Ablation[{name}] RMSE: {rmse}")
        return rmse

    data = skrub.var("data", train_subset)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    if use_vectorizer:
        vectorizer = skrub.TableVectorizer()
        X_transformed = X.skb.apply(vectorizer)
    else:
        # Structural ablation: use raw numeric columns only, if any
        X_transformed = X

    model = HistGradientBoostingRegressor(random_state=0)
    pred_graph = X_transformed.skb.apply(model, y=y)
    learner = pred_graph.skb.make_learner(fitted=True)

    val_features = val_subset.drop(columns=[target_col], errors="ignore")
    val_pred = learner.predict({"data": val_features})
    rmse = mean_squared_error(val_subset[target_col], val_pred) ** 0.5
    print(f"Ablation[{name}] RMSE: {rmse}")
    return rmse


results = {}

# Main reference pipeline
results["reference_full"] = run_variant(
    "reference_full",
    use_vectorizer=True,
    use_target=True,
)

# Ablation 1: remove TableVectorizer
results["no_vectorizer"] = run_variant(
    "no_vectorizer",
    use_vectorizer=False,
    use_target=True,
)

# Ablation 2: remove supervised model path, use mean baseline
results["mean_baseline"] = run_variant(
    "mean_baseline",
    use_vectorizer=True,
    use_target=False,
)

best_variant = min(results, key=results.get)
best_score = results[best_variant]

print(f"Final Validation Performance: {results['reference_full']}")
print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")

# Contribution analysis: compare each ablation against the full reference
full_score = results["reference_full"]
deltas = {k: v - full_score for k, v in results.items() if k != "reference_full"}
most_important = max(deltas, key=lambda k: abs(deltas[k]))
print("Ablation impact relative to reference (positive means worse RMSE):")
for k, delta in deltas.items():
    print(f"  {k}: {delta}")

print(
    f"Part contributing the most to overall performance: {most_important} "
    f"(RMSE change: {deltas[most_important]})"
)