
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
from skrub import ApplyToCols, SquashingScaler

# Paths
INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# -------------------------
# Pipeline 1: original
# -------------------------
def build_graph_original(dataframe):
    data_train = skrub.var("data", dataframe)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        random_seed=42,
        verbose=0,
    )

    pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    return pred_graph

# -------------------------
# Pipeline 2: ablation with guarded ratios + squashing
# -------------------------
@skrub.deferred
def add_guarded_housing_ratios(df):
    out = df.copy()

    if "total_rooms" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household_struct"] = (
            (out["total_rooms"] / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    if "population" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household_struct"] = (
            (out["population"] / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return out


def build_graph_ablation(dataframe, use_derived_features):
    data = skrub.var("data", dataframe)

    if use_derived_features:
        data = data.skb.apply_func(add_guarded_housing_ratios)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    X = X.skb.apply(ApplyToCols(SquashingScaler(max_absolute_value=3), cols=skrub.selectors.numeric()))

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        random_seed=42,
        verbose=0,
    )

    pred_graph = X.skb.apply(vectorizer).skb.apply(model, y=y)
    return pred_graph


# -------------------------
# Holdout evaluation for both pipelines
# -------------------------
pred_graph_1 = build_graph_original(train_part)
learner_1 = pred_graph_1.skb.make_learner(fitted=True)
valid_pred_1 = np.asarray(learner_1.predict({"data": valid_part})).reshape(-1)

pred_graph_2 = build_graph_ablation(train_part, use_derived_features=True)
learner_2 = pred_graph_2.skb.make_learner(fitted=True)
valid_pred_2 = np.asarray(learner_2.predict({"data": valid_part})).reshape(-1)

# Tiny 1D weight search on holdout
weights = np.arange(0.0, 1.0001, 0.05)
best_w = 0.5
best_score = np.inf

for w in weights:
    blended = w * valid_pred_1 + (1.0 - w) * valid_pred_2
    score = mean_squared_error(valid_part[target_col], blended) ** 0.5
    if score < best_score:
        best_score = score
        best_w = float(w)

# Rank-averaging safety blend
rank_pred_1 = rankdata(valid_pred_1, method="average")
rank_pred_2 = rankdata(valid_pred_2, method="average")
rank_blend = 0.5 * rank_pred_1 + 0.5 * rank_pred_2
rank_weight = 0.5

# Compare raw ensemble and rank ensemble on holdout
raw_blend_valid = best_w * valid_pred_1 + (1.0 - best_w) * valid_pred_2
raw_score = mean_squared_error(valid_part[target_col], raw_blend_valid) ** 0.5

rank_blended_valid = rank_weight * rank_blend + (1.0 - rank_weight) * raw_blend_valid
rank_score = mean_squared_error(valid_part[target_col], rank_blended_valid) ** 0.5

if rank_score < raw_score:
    final_validation_score = rank_score
    use_rank_blend = True
else:
    final_validation_score = raw_score
    use_rank_blend = False

print(f"Ablation[original] RMSE: {mean_squared_error(valid_part[target_col], valid_pred_1) ** 0.5}")
print(f"Ablation[derived_features] RMSE: {mean_squared_error(valid_part[target_col], valid_pred_2) ** 0.5}")
print(f"Best ensemble weight (pred_1): {best_w}")
print(f"Final Validation Performance: {final_validation_score}")

# -------------------------
# Final submission stage
# -------------------------
# Train each solution on the full training data using its existing graph
full_pred_graph_1 = build_graph_original(train_df)
full_learner_1 = full_pred_graph_1.skb.make_learner(fitted=True)

full_pred_graph_2 = build_graph_ablation(train_df, use_derived_features=True)
full_learner_2 = full_pred_graph_2.skb.make_learner(fitted=True)

test_pred_1 = np.asarray(full_learner_1.predict({"data": test_df})).reshape(-1)
test_pred_2 = np.asarray(full_learner_2.predict({"data": test_df})).reshape(-1)

raw_test_blend = best_w * test_pred_1 + (1.0 - best_w) * test_pred_2

if use_rank_blend:
    test_rank_1 = rankdata(test_pred_1, method="average")
    test_rank_2 = rankdata(test_pred_2, method="average")
    test_rank_blend = 0.5 * test_rank_1 + 0.5 * test_rank_2
    test_pred = 0.5 * test_rank_blend + 0.5 * raw_test_blend
else:
    test_pred = raw_test_blend

# Write submission
if os.path.exists(sample_path):
    submission = pd.read_csv(sample_path)
    target_submission_col = submission.columns[-1]
    submission[target_submission_col] = test_pred
else:
    submission = pd.DataFrame({target_col: test_pred})

submission.to_csv("submission.csv", index=False)
