# Multi-table DataOps pattern (groupby + merge)

This reference is based on the local `1120_multiple_tables` DataOps example.
Use it when features and targets come from related tables.

## Verified APIs (from local docs)
- `skrub.var(...)`
- `.skb.mark_as_X()`, `.skb.mark_as_y()`
- Dataframe operations inside DataOps expressions (`assign`, `groupby`, `agg`, `merge`, `drop`)
- `.skb.apply(...)` with `TableVectorizer` and estimator
- `.skb.make_randomized_search(...)`

## Canonical pattern
```python
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

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
    HistGradientBoostingClassifier(
        learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
    ),
    y=fraud_flags,
)
```

## Checklist
- Keep table linkage logic in the DataOps graph, not external disconnected preprocessing.
- Mark `X` and `y` on the correct base table columns before model application.
- Aggregate child-table features to the prediction unit before final estimator.
