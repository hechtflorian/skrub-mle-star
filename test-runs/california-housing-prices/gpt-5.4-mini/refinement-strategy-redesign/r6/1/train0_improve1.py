

import pandas as pd
import numpy as np
import skrub
from skrub import ApplyToCols, selectors as s
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TARGET = "median_house_value"
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Hold-out validation split
train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

def make_learner_from_features(X_train_df, y_train):
    data = skrub.var("data", X_train_df)
    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    low_card_sel = s.string() & s.cardinality_below(20)
    high_card_sel = s.string() & ~s.cardinality_below(20)

    # Baseline: single global vectorizer
    baseline_vec = skrub.TableVectorizer()
    baseline_X = X.skb.apply(baseline_vec)
    baseline_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=1800,
        random_seed=42,
        verbose=0,
    )
    baseline_graph = baseline_X.skb.apply(baseline_model, y=y)
    baseline_learner = baseline_graph.skb.make_learner(fitted=True)

    # Selector-driven split:
    # - low-cardinality strings -> preserve as categorical
    # - high-cardinality strings -> lighter, stable encoder
    # - everything else -> default table vectorization
    low_cat = ApplyToCols(skrub.ToCategorical(), cols=low_card_sel)
    high_enc = ApplyToCols(skrub.MinHashEncoder(n_components=16), cols=high_card_sel)
    routed_vec = skrub.TableVectorizer(
        low_cardinality=skrub.ToCategorical(),
        high_cardinality=skrub.MinHashEncoder(n_components=16),
    )

    routed_parts = [
        X.skb.select(~(low_card_sel | high_card_sel)).skb.apply(routed_vec),
        X.skb.select(low_card_sel).skb.apply(low_cat),
        X.skb.select(high_card_sel).skb.apply(high_enc),
    ]
    routed_X = routed_parts[0].skb.concat(routed_parts[1:], axis=1)

    routed_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=1800,
        random_seed=42,
        verbose=0,
    )
    routed_graph = routed_X.skb.apply(routed_model, y=y)
    routed_learner = routed_graph.skb.make_learner(fitted=True)

    # Cheap ablation on holdout validation
    baseline_valid_pred = baseline_learner.predict({"data": valid_part})
    baseline_rmse = mean_squared_error(valid_part[TARGET], baseline_valid_pred) ** 0.5
    print(f"Ablation[baseline_table_vectorizer] RMSE: {baseline_rmse}")

    routed_valid_pred = routed_learner.predict({"data": valid_part})
    routed_rmse = mean_squared_error(valid_part[TARGET], routed_valid_pred) ** 0.5
    print(f"Ablation[selector_routed] RMSE: {routed_rmse}")

    if routed_rmse <= baseline_rmse:
        print(f"Best ablation variant: selector_routed | RMSE: {routed_rmse}")
        return "routed"
    else:
        print(f"Best ablation variant: baseline_table_vectorizer | RMSE: {baseline_rmse}")
        return "baseline"

best_variant = make_learner_from_features(train_part, train_part[TARGET])

# Fit on full data for test prediction with chosen fixed configuration
full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
full_y = full_data[TARGET].skb.mark_as_y()

low_card_sel = s.string() & s.cardinality_below(20)
high_card_sel = s.string() & ~s.cardinality_below(20)

if best_variant == "baseline":
    full_vec = skrub.TableVectorizer()
    full_X_vec = full_X.skb.apply(full_vec)
else:
    low_cat = ApplyToCols(skrub.ToCategorical(), cols=low_card_sel)
    high_enc = ApplyToCols(skrub.MinHashEncoder(n_components=16), cols=high_card_sel)
    routed_vec = skrub.TableVectorizer(
        low_cardinality=skrub.ToCategorical(),
        high_cardinality=skrub.MinHashEncoder(n_components=16),
    )
    full_parts = [
        full_X.skb.select(~(low_card_sel | high_card_sel)).skb.apply(routed_vec),
        full_X.skb.select(low_card_sel).skb.apply(low_cat),
        full_X.skb.select(high_card_sel).skb.apply(high_enc),
    ]
    full_X_vec = full_parts[0].skb.concat(full_parts[1:], axis=1)

final_model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.05,
    iterations=1800,
    random_seed=42,
    verbose=0,
)

full_pred_graph = full_X_vec.skb.apply(final_model, y=full_y)
full_learner = full_pred_graph.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
