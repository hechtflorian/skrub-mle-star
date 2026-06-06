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

## Minimum DataOps pipeline shape
```python
import skrub

X = skrub.X(train_df.drop(columns=target_col, errors="ignore"))
y = skrub.y(train_df[target_col])

n_components = skrub.choose_int(5, 15, name="n_components")
encoder = skrub.TableVectorizer(
    high_cardinality=skrub.choose_from(
        {
            "minhash": skrub.MinHashEncoder(n_components=n_components),
            "lsa": skrub.StringEncoder(n_components=n_components),
        },
        name="encoder",
    )
)

clf = YourModel(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)

pred = X.skb.apply(encoder).skb.apply(clf, y=y)
```

## Safe execution pattern (fit/predict without API misuse)
```python
# Option A: evaluate directly from DataOp
cv_results = pred.skb.cross_validate()

# Option B: compile learner and predict with environment dict
learner = pred.skb.make_learner(fitted=True)
pred_test = learner.predict({"data": test_df})

# Avoid redundant target drops at inference time:
# do not call test_df.drop(columns=target_col) unless truly needed.
```

## Never do this
- `pred.fit(X, y)` where `pred` is a DataOps expression.
- `learner.predict(test_data_op)` with a raw DataOp node.
- `mean_squared_error(..., squared=False)` in this runtime.

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
    scoring="roc_auc", n_iter=8, n_jobs=4, random_state=0, fitted=True
)
```

## Validation checklist
- Pipeline is DataOps-first (`.skb.apply(...)` main path).
- `X`/`y` are explicitly marked.
- Tunables are embedded with `choose_*`/`choose_from`.
- Search runs from DataOp (`make_randomized_search` or `make_grid_search`).
- Prediction uses dict environments keyed by source variable names.

## When to load other references
- Load `choices_hparam_pattern.md` when adding `skrub.choose_*` / `skrub.choose_from(...)` or randomized/grid search for hyperparameter tuning.
- Load `dataops_tuning_optuna.md` when using Optuna backend or trial-based search flows for tuning.
- Load `encoding_skrub.md` when changing feature encoding, preprocessing, or selector-based routing.
- Load `joining_across_columns.md` for multi-table merge/aggregation pipelines.
- Load `common_failure_fixes.md` when runtime errors appear or metric parsing fails.
- Load `skrub_subsampling.md` when iteration speed is the bottleneck and subsampling is required.