# Encoding with `TableVectorizer`

Use this reference for feature encoding before supervised modeling.

## Verified APIs (from local docs)
- `skrub.tabular_pipeline("regressor")`
- `skrub.TableVectorizer(...)`
- `sklearn.pipeline.make_pipeline(...)`
- Optional encoder choices used in examples: `MinHashEncoder`, `GapEncoder`, `ToCategorical`

## Encoder behavior summary
`TableVectorizer` handles heterogeneous columns by type:
- numeric -> numeric cast
- datetime -> datetime feature extraction
- low-cardinality categorical -> one-hot style encoding
- high-cardinality categorical/string -> text/category encoding (default uses `GapEncoder`)

## Pattern 1: fastest baseline
```python
import pandas as pd
from sklearn.model_selection import cross_validate
from skrub import tabular_pipeline
from skrub.datasets import fetch_employee_salaries

employees = pd.read_csv(fetch_employee_salaries().path)
X = employees.drop(columns="current_annual_salary")
y = employees["current_annual_salary"]

model = tabular_pipeline("regressor")
results = cross_validate(model, X, y)
```

## Pattern 2: explicit vectorizer + estimator
```python
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())
pipeline.fit(X, y)
```

## Pattern 3: tree-focused vectorizer specialization
```python
from skrub import MinHashEncoder, TableVectorizer, ToCategorical
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

vectorizer = TableVectorizer(
    low_cardinality=ToCategorical(),
    high_cardinality=MinHashEncoder(),
)
pipeline = make_pipeline(
    vectorizer,
    HistGradientBoostingRegressor(categorical_features="from_dtype"),
)
```

## Checklist
- Prefer `TableVectorizer` over manual pandas encoding chains.
- Keep the preprocessing and estimator steps in one reproducible pipeline.
- Choose vectorizer variants based on downstream estimator constraints.