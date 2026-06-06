import os
import warnings

import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TARGET = "median_house_value"
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

train_df = pd.read_csv(TRAIN_PATH)

# Keep the subsampling behavior from the original solution.
train_df = train_df.sample(n=min(len(train_df), len(train_df)), random_state=42).reset_index(drop=True)

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def evaluate_variant(name, use_vectorizer=True, use_model=True):
    data = skrub.var("data", train_part)
    X = data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    if use_vectorizer:
        X_proc = X.skb.apply(skrub.TableVectorizer())
    else:
        X_proc = X

    if use_model:
        pred = X_proc.skb.apply(HistGradientBoostingRegressor(random_state=42), y=y)
    else:
        # Simple ablation: disable the model stage by using a naive constant predictor
        # implemented outside the DataOps chain for comparison.
        pred = None

    if use_model:
        learner = pred.skb.make_learner(fitted=True)
        valid_features = valid_part.drop(columns=[TARGET], errors="ignore")
        valid_preds = learner.predict({"data": valid_features})
    else:
        valid_preds = [train_part[TARGET].mean()] * len(valid_part)

    score = rmse(valid_part[TARGET], valid_preds)
    print(f"Ablation[{name}] RMSE: {score}")
    return score

# Baseline: original pipeline structure
baseline_score = evaluate_variant("baseline", use_vectorizer=True, use_model=True)

# Ablation 1: remove TableVectorizer, feed raw numeric columns directly to the model
no_vectorizer_score = evaluate_variant("no_vectorizer", use_vectorizer=False, use_model=True)

# Ablation 2: replace learned model with a simple mean baseline
mean_baseline_score = evaluate_variant("mean_baseline", use_vectorizer=False, use_model=False)

results = {
    "baseline": baseline_score,
    "no_vectorizer": no_vectorizer_score,
    "mean_baseline": mean_baseline_score,
}

best_variant = min(results, key=results.get)
best_score = results[best_variant]

print(f"Final Validation Performance: {baseline_score}")
print("Ablation summary:")
for k, v in results.items():
    print(f"  {k}: RMSE={v}")

print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")

# Which part contributes most?
improvement_no_vectorizer = no_vectorizer_score - baseline_score
improvement_mean_baseline = mean_baseline_score - baseline_score

contribs = {
    "TableVectorizer": improvement_no_vectorizer,
    "HistGradientBoostingRegressor": improvement_mean_baseline,
}

most_important = max(contribs, key=lambda k: contribs[k])
print(
    f"Most important part contributing to performance: {most_important} "
    f"(largest RMSE degradation when removed: {contribs[most_important]})"
)