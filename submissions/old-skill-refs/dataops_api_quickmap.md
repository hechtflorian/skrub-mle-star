# Skrub DataOps: Building multi-table ML pipelines with skrub DataOps

High-level reference for building declarative tabular data preprocessing pipelines, relational joins, and DataOps graphs using the `skrub` library. `skrub` is a Python library built to seamlessly bridge tabular data (like pandas dataframes) to scikit-learn machine learning pipelines.

## Agent Verification Checklist (Skrub)
Use this checklist when generating data preprocessing code:
- [ ] **No Pandas Engineering**: Did you replace manual `pd.get_dummies()`, `.fillna()`, and `.apply()` with `skrub.TableVectorizer`?
- [ ] **DataOps Graph**: If building a tunable model, did you define inputs using `skrub.var()` or `skrub.X()`, `skrub.y()` and export via `.skb.make_learner()`?
- [ ] **Relational Data**: If given multiple tables, did you use `skrub.Joiner` or `skrub.AggJoiner` instead of `pd.merge()`?

## Quick Reference
- **Core Pipeline**: `skrub.tabular_pipeline(estimator)`
- **Auto-Encoder**: `TableVectorizer(high_cardinality="minhash")`
- **Fuzzy Joining**: `Joiner(aux_table, main_key="id", aux_key="id")`
- **DataOps Tuning**: `skrub.choose_from(["minhash", "one_hot"], name="encoder")`

---

## 1. DataOps Graph (Building & Tuning)
Use these tools to build complex, declarative, and hyperparameter-tunable DAGs.
* **`skrub.var(name: str, Optional[value: object])`**: Create a skrub variable, representing inputs to DataOps plan and the corresponding learner.
* **`skrub.DataOp.skb.mark_as_X()`**: Mark this DataOp as being the X table.
* **`skrub.DataOp.skb.make_learner()`**: Compiles the DataOps operations into a standard scikit-learn estimator.
* **`skrub.choose_from(outcomes, name="...")`**: Defines a categorical hyperparameter search space directly inline.
* **`skrub.choose_int(lower, upper)`**: Defines an integer hyperparameter search space.
* **`skrub.choose_float(lower, upper)`**: Defines a float hyperparameter search space.


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
