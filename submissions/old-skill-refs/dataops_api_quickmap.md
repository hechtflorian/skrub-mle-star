# Skrub DataOps: Building multi-table ML pipelines with skrub DataOps

High-level reference for building declarative tabular data preprocessing pipelines, relational joins, and DataOps graphs using the `skrub` library. `skrub` is a Python library built to seamlessly bridge tabular data (like pandas dataframes) to scikit-learn machine learning pipelines.

## Agent Verification Checklist (Skrub)
Use this checklist when generating data preprocessing code:
- [ ] **No Pandas Engineering**: Did you use `skrub.TableVectorizer` to transform the dataframe to a vectorized representation?
- [ ] **DataOps Graph**: Did you define inputs using `skrub.X()`, `skrub.y()` or `skrub.var()` and `skrub.DataOp.skb.mark_as_X()`, `skrub.DataOp.skb.mark_as_y()`?
- [ ] **Relational Data**: If given multiple tables, did you use `skrub.Joiner` or `skrub.AggJoiner` instead of `pd.merge()`?

---

## 1. DataOps
Use these tools to build complex, declarative, and hyperparameter-tunable DAGs.

### 1.1 Generalizing scikit-learn pipeline:
* **`skrub.var()`**: Create a skrub variable.
* **`skrub.X()`**: Create a skrub variable and mark it as being X (shortcut for `.skb.mark_as_X()`)
* **`skrub.y()`**: Create a skrub variable and mark it as being y (shortcut for `.skb.mark_as_y()`)
* **`skrub.as_data_op()`**: Create a DataOp that evaluates to the given value, wraps any object.
* **`skrub.deferred()`**: Wrap function calls in a DataOp, call is executed when DataOp is evaluated.

### 1.2 Inline hyperparameter selection in DataOps plan:
* **`skrub.choose_bool()`**: Choice between `True` and `False`.
* **`skrub.choose_float()`**: Choice of floating-point numbers from a numeric range.
* **`skrub.choose_int()`**: Choice of integers from a numeric range.
* **`skrub.choose_from()`**: Choice among several possible outcomes.
* **`skrub.optional()`**: Choice between `value` and `None`.

### 1.3 Evaluate DataOps plan:
* **`skrub.cross_validate()`**: Cross-validate a learner built from a DataOp.
* **`skrub.eval_mode()`**: Return the mode in which the DataOp is currently being evaluated.

### 1.4 The `skb` accessor exposes all DataOps methods and attributes:
* **`skrub.DataOp.skb.apply()`**: Apply an estimator that follows the scikit-learn API to a dataframe or numpy array.
* **`skrub.DataOp.skb.apply_func()`**: Apply the given function.
* **`skrub.DataOp.skb.clone()`**: Get an independent clone of the DataOp.
* **`skrub.DataOp.skb.concat()`**: Concatenate dataframes vertically or horizontally.
* **`skrub.DataOp.skb.cross_validate()`**: Cross-validate the DataOp plan.
* **`skrub.DataOp.skb.describe_defaults()`**: Describe the hyper-parameters used by the default learner.
* **`skrub.DataOp.skb.describe_param_grid()`**: Describe the hyper-parameters extracted from choices in the DataOp.
* **`skrub.DataOp.skb.describe_steps()`**: Get a text representation of the computation graph.
* **`skrub.DataOp.skb.draw_graph()`**: Get an SVG string representing the computation graph.
* **`skrub.DataOp.skb.drop()`**: Drop some columns.
* **`skrub.DataOp.skb.eval()`**: Evaluate the DataOp.
* **`skrub.DataOp.skb.freeze_after_fit()`**: Freeze the result during learner fitting.
* **`skrub.DataOp.skb.full_report()`**: Generate a full report of the DataOp's evaluation.
* **`skrub.DataOp.skb.get_data()`**: Collect the values of the variables contained in the DataOp.
* **`skrub.DataOp.skb.get_vars()`**: Get all the variables used in the DataOp.
* **`skrub.DataOp.skb.make_learner()`**: Get a skrub learner for this DataOp.
* **`skrub.DataOp.skb.make_grid_search()`**: Find the best parameters with grid search.
* **`skrub.DataOp.skb.make_randomized_search()`**: Find the best parameters with randomized search.
* **`skrub.DataOp.skb.if_else()`**: Create a conditional DataOp.
* **`skrub.DataOp.skb.iter_cv_splits()`**: Yield splits of an environment into training and testing environments.
* **`skrub.DataOp.skb.iter_learners_randomized()`**: Get learners with different parameter combinations.
* **`skrub.DataOp.skb.mark_as_X()`**: Mark this DataOp as being the X table.
* **`skrub.DataOp.skb.mark_as_y()`**: Mark this DataOp as being the y table.
* **`skrub.DataOp.skb.match()`**: Select based on the value of a DataOp.
* **`skrub.DataOp.skb.preview()`**: Get the value computed for previews (shown when printing the DataOp).
* **`skrub.DataOp.skb.subsample()`**: Configure subsampling of a dataframe or numpy array.
* **`skrub.DataOp.skb.train_test_split()`**: Split an environment into training and testing environments.
* **`skrub.DataOp.skb.with_scoring()`**: Attach a scoring method to this DataOp.
* **`skrub.DataOp.skb.find()`**: Find a node (DataOp or choice) in the computational graph.
* **`skrub.DataOp.skb.find_X_y()`**: Find the nodes that have been marked with `mark_as_X()` and `mark_as_y()`.
* **`skrub.DataOp.skb.applied_estimator()`**: Retrieve the estimator applied in the previous step, as a DataOp.



## 2. Example multi-table ML pipeline using skrub DataOps
```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import skrub
import skrub.datasets   # for example datasets

train_data = skrub.datasets.fetch_credit_fraud(split="train")

# Define skrub vars for inputs to the DataOps plan to be built
baskets = skrub.var("baskets", pd.read_csv(train_data.baskets_path))
products = skrub.var("products", pd.read_csv(train_data.products_path))

# Mark X and y to allow DataOps to track indices of vars
basket_ids = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# Filter products table to keep only those that match one of the baskets in basket table
kept_products = products[products["basket_ID"].isin(basket_ids["ID"])]
products_with_total = kept_products.assign(
    total_price=kept_products["Nbr_of_prod_purchas"] * kept_products["cash_price"]
)

# Build skrub.TableVectorizer with different choices of type of encoder and components
n = skrub.choose_int(5, 15, name="n_components")
encoder = skrub.choose_from(
    {
        "MinHash": skrub.MinHashEncoder(n_components=n),
        "LSA": skrub.StringEncoder(n_components=n),
    },
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)

# Restrict vectorizer to subset of columns
vectorized_products = products_with_total.skb.apply(
    vectorizer, exclude_cols="basket_ID"
)

# Aggregate vectorized products by basked ID, and merge result with basket table
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
augmented_baskets = basket_ids.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])

# Add supervised estimator, and use skrub.choose_float() to add LR as tunable hyperparam
hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)
predictions = augmented_baskets.skb.apply(hgb, y=fraud_flags)

# Can use make_randomized_search() to perform hyperparam tuning
search = predictions.skb.make_randomized_search(
    scoring="roc_auc", n_iter=8, n_jobs=4, random_state=0, fitted=True
)

# Get best performing SkrubLearner and use for inference on test data
test_data = skrub.datasets.fetch_credit_fraud(split="test")

new_baskets = pd.read_csv(test_data.baskets_path)
new_products = pd.read_csv(test_data.products_path)

probabilities = search.best_learner_.predict_proba(
    {"baskets": new_baskets, "products": new_products}
)
```

## When to load deeper references
- Multi-table joins/aggregations/entity relationships: load `multi_table_pipeline_pattern.md`.
- Choice/tuning logic and search-space composition: load `choices_hparam_pattern.md`.
- Runtime exceptions, shape/type mismatches, unresolved symbols: load `common_failure_fixes.md`.

## Web verification pattern (when uncertain)
Use targeted searches and patch only the uncertain line:
- `site:skrub-data.org <symbol_name>`
- `site:skrub-data.org DataOps <symbol_name>`
- `site:skrub-data.org reference data_ops`
