# Skrub DataOps Quickmap

Use this file as the canonical reference for DataOps-first pipeline structure.

## Essential API calls
- Variables: `skrub.var(...)`, `skrub.X(...)`, `skrub.y(...)`
- Marking roles: `.skb.mark_as_X()`, `.skb.mark_as_y()` (or use `skrub.X(...)` and `skrub.y(...)`)
- Chaining: `.skb.apply(transformer_or_estimator, ...)`
- Tuning choices: `skrub.choose_int(...)`, `skrub.choose_float(...)`, `skrub.choose_from(...)`
- Search: `.skb.make_randomized_search(...)`, `.skb.make_grid_search(...)`
- Learner: `.skb.make_learner(...)`
- Diagnostics: `.skb.describe_param_grid()`

## `skrub.choose_from(...)` key-type safety
- When using dictionary-form `choose_from`, keys are names and must be strings.
- Valid: `skrub.choose_from({"small": 6, "medium": 8}, name="max_depth")`
- Invalid: `skrub.choose_from({6: 6, 8: 8}, name="max_depth")`
- Numeric range fallback: `skrub.choose_int(low, high, name="...")` or `skrub.choose_float(low, high, name="...")`

## Minimum DataOps pipeline shape (example)
```python
import skrub

X = skrub.X(df.drop(columns=target_col, errors="ignore"))
y = skrub.y(df[target_col])

n_components = skrub.choose_int(5, 15, name="n_components")
encoder = skrub.TableVectorizer(
    high_cardinality=skrub.choose_from(
        {
            "minhash": skrub.MinHashEncoder(n_components=n_components), # your_encoder_1
            "lsa": skrub.StringEncoder(n_components=n_components),  # your_encoder_2
        },
        name="encoder",
    )
)

model = YourModel(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)

pred = X.skb.apply(encoder).skb.apply(model, y=y)
```

## Holdout validation (default for init / ablation / refinement / tuning)
`skrub.var("data", df)` sets which rows `make_learner(fitted=True)` fits on. Predicting on `valid_part` after fitting on full `train_df` **leaks** validation rows into training and inflates scores.

**Anti-pattern (leakage):**
```python
data = skrub.var("data", train_df)  # full train bound
# ... split train_part / valid_part ...
learner = pred.skb.make_learner(fitted=True)  # trained on ALL rows including valid_part
valid_pred = learner.predict({"data": valid_part})  # optimistic validation score
```

**Default script (early stages — holdout metric only):**
```python
import numpy as np
from sklearn.model_selection import train_test_split
# Use the competition metric from task_description.txt (# Metric section)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()
pred = X_train.skb.apply(encoder).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
holdout_score = your_metric_fn(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {holdout_score}")
```

Rules:
- Early stages: bind **`train_part`**, print holdout metric, then **stop** — no `test_df`, no full-train refit, no `submission.csv`.
- Submission export: load `references/submission_export.md` in the submission agent only.
- Keep the same split (`test_size`, `random_state`) across stages.
- Preprocessing inside `.skb.apply_func` / transformers learns from bound rows only — binding `train_part` prevents val/test rows from influencing fit-time stats.
- Tuning search: `search.fit({"data": train_part})`, eval with `search.best_learner_.predict({"data": valid_part})` — see `choices_hparam_pattern.md`.

## Safe execution pattern (fit/predict without API misuse)
```python
# Option A: cross-validation from DataOp (when CV is appropriate)
cv_results = pred.skb.cross_validate()

# Option B: holdout metric — bind train_part (default for early stages)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

# Avoid redundant target drops at inference time:
# do not call test_df.drop(columns=target_col) unless truly needed.
```

## Never do this
- `pred.fit(X, y)` where `pred` is a DataOps expression.
- `learner.predict(test_data_op)` with a raw DataOp node.
- `mean_squared_error(..., squared=False)` in this runtime (use `** 0.5` for RMSE tasks instead).

## Multi-table pattern
```python
import pandas as pd
import skrub

dataset = skrub.datasets.fetch_credit_fraud(split="train")
baskets = skrub.var("baskets", pd.read_csv(dataset.baskets_path))
products = skrub.var("products", pd.read_csv(dataset.products_path))

basket_ids = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

kept_products = products[products["basket_ID"].isin(basket_ids["ID"])]
products_with_total = kept_products.assign(
    total_price=kept_products["Nbr_of_prod_purchas"] * kept_products["cash_price"]
)

n = skrub.choose_int(5, 15, name="n_components")
encoder = skrub.choose_from(
    {
        "MinHash": skrub.MinHashEncoder(n_components=n),
        "LSA": skrub.StringEncoder(n_components=n),
    },
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)
vectorized_products = products_with_total.skb.apply(vectorizer, exclude_cols="basket_ID")

aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
augmented_baskets = basket_ids.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])

pred = augmented_baskets.skb.apply(
    YourModel(
        learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
    ),
    y=fraud_flags,
)
search = pred.skb.make_randomized_search(
    scoring="scoring_func", n_iter=n_iter, n_jobs=n_jobs, random_state=random_state, fitted=True
)  # tree boosters: default search n_jobs=1; or estimator n_jobs=1 + search n_jobs=2; max 4 for very fast single-threaded fits
```

## Validation checklist
- Pipeline is DataOps-first (`.skb.apply(...)` main path).
- `X`/`y` are explicitly marked.
- Early stages: holdout metric only (`train_part` bind + `valid_part` predict); no `test_df`, full-train refit, or `submission.csv`.
- Submission export: `references/submission_export.md` (submission agent only).
- Tunables are embedded with `choose_*`/`choose_from`.
- Search runs from DataOp (`make_randomized_search` or `make_grid_search`).
- Prediction uses dict environments keyed by source variable names.

## When to load other references
- Load `choices_hparam_pattern.md` when adding `skrub.choose_*` / `skrub.choose_from(...)` or randomized/grid search for hyperparameter tuning.
- Load `encoding_skrub.md` when changing feature encoding, preprocessing, or selector-based routing.
- Load `joining_across_columns.md` for multi-table merge/aggregation pipelines.
- Load `common_failure_fixes.md` when runtime errors appear or metric parsing fails.
- Load `ensemble_dataops_patterns.md` when merging multiple model legs (holdout metric only).
- Load `submission_export.md` in the **submission agent only** for full-train refit + test export.
- Load `skrub_subsampling.md` when iteration speed is the bottleneck and subsampling is required.