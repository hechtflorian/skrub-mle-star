

import os
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "median_house_value"

# Keep the DataOps graph intact
data = skrub.var("data", train)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# Freeze/simplify the vectorizer choice, but keep the pipeline structure.
# Use a slightly richer encoder config if helpful, while prioritizing model-side tuning.
vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.MinHashEncoder(
        n_components=skrub.choose_int(8, 24, name="high_card_n_components")
    )
)

# Widen HGB search a bit: depth matters most, then leaf and learning rate.
model = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.01, 0.2, log=True, name="learning_rate"),
    max_depth=skrub.choose_int(3, 12, name="max_depth"),
    min_samples_leaf=skrub.choose_int(5, 60, name="min_samples_leaf"),
    l2_regularization=skrub.choose_float(1e-6, 1e-1, log=True, name="l2_regularization"),
    random_state=42,
)

pred = X.skb.apply(vectorizer).skb.apply(model, y=y)

# DataOps-native randomized search over the in-graph choices.
# Keep it small and fast, but search the model-side knobs rather than hand-picking.
search = pred.skb.make_randomized_search(
    n_iter=12,
    random_state=42,
    n_jobs=1,
    fitted=True,
)
trained_learner = search.best_learner_

# Single stability check with a second cheap contiguous split.
n = len(train)
val_size = max(1, int(0.2 * n))

# Main split
main_train_data = train.iloc[:-val_size].reset_index(drop=True)
main_val_data = train.iloc[-val_size:].reset_index(drop=True)

main_graph = skrub.var("data", main_train_data)
X_main = main_graph.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_main = main_graph[target_col].skb.mark_as_y()

main_pred_graph = X_main.skb.apply(vectorizer).skb.apply(model, y=y_main)
main_search = main_pred_graph.skb.make_randomized_search(
    n_iter=12,
    random_state=42,
    n_jobs=1,
    fitted=True,
)
main_learner = main_search.best_learner_

main_val_pred = main_learner.predict({"data": main_val_data})
main_val_rmse = mean_squared_error(main_val_data[target_col], main_val_pred) ** 0.5
print(f"Final Validation Performance: {main_val_rmse}")

# Secondary split for cheap stability check.
second_val_start = max(1, n - 2 * val_size)
second_train_data = train.iloc[:second_val_start].reset_index(drop=True)
second_val_data = train.iloc[second_val_start:].reset_index(drop=True)

if len(second_train_data) > 0 and len(second_val_data) > 0:
    second_graph = skrub.var("data", second_train_data)
    X_second = second_graph.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_second = second_graph[target_col].skb.mark_as_y()

    second_pred_graph = X_second.skb.apply(vectorizer).skb.apply(model, y=y_second)
    second_search = second_pred_graph.skb.make_randomized_search(
        n_iter=8,
        random_state=42,
        n_jobs=1,
        fitted=True,
    )
    second_learner = second_search.best_learner_

    second_val_pred = second_learner.predict({"data": second_val_data})
    second_val_rmse = mean_squared_error(second_val_data[target_col], second_val_pred) ** 0.5
    print(f"Secondary Validation Performance: {second_val_rmse}")

# Final fit on full data for submission
final_learner = pred.skb.make_randomized_search(
    n_iter=12,
    random_state=42,
    n_jobs=1,
    fitted=True,
).best_learner_

test_pred = final_learner.predict({"data": test})
submission = pd.DataFrame({target_col: test_pred})
submission.to_csv("submission.csv", index=False)
