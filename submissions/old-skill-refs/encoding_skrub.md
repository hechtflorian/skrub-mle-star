# Encoding: from a dataframe to a numerical matrix for ML

This example shows how to transform a rich dataframe with columns of various types into a numerical matrix on which machine-learning algorithms can be applied. This example explores `skrub.TableVectorizer`: The TableVectorizer bridges the gap between tabular data and machine-learning pipelines. It allows us to apply a machine-learning estimator to our dataframe without manual data wrangling and feature extraction. 

The TableVectorizer distinguishes between 4 basic kinds of columns. For each kind, it applies a different transformation, which we can configure. The kinds of columns and the default transformation for each of them are:
- numeric columns: simply casting to floating-point
- datetime columns: extracting features such as year, day, hour with the `DatetimeEncoder`
- low-cardinality categorical columns: one-hot encoding
- high-cardinality categorical columns: a simple and effective text representation pipeline provided by the `GapEncoder`



## Agent Verification Checklist (Skrub) - MODIFY THIS
Use this checklist when generating data preprocessing code:
- [ ] **No Pandas Engineering**: Did you use `skrub.TableVectorizer` to transform the dataframe to a vectorized representation?
- [ ] **DataOps Graph**: Did you define inputs using `skrub.X()`, `skrub.y()` or `skrub.var()` and `skrub.DataOp.skb.mark_as_X()`, `skrub.DataOp.skb.mark_as_y()`?
- [ ] **Relational Data**: If given multiple tables, did you use `skrub.Joiner` or `skrub.AggJoiner` instead of `pd.merge()`?

---

## 1. Basic example for preparing a dataframe and encoding features for ML:
```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate

import skrub
from skrub import tabular_pipeline
import skrub.datasets


file_path = skrub.datasets.fetch_employee_salaries().path
employees = pd.read_csv(file_path)  # heterogeneous set of columns
X = employees.drop(columns="current_annual_salary")
y = employees["current_annual_salary"]

# Vectorize table
# Estimator returned by `tabluar_pipeline` uses `skrub.TableVectorizer` to preprocess dataframe and vectorize features, and includes a supervised learner (default=`HistGradientBoostingRegressor`)
model = tabular_pipeline("regressor")   # or other estimator : str = "regression", "classifier", "classification"
results = cross_validate(model, X, y)
```

## 2. More details on encoding tabular data
```python
from sklearn.ensemble import HistGradientBoostingRegressor

from skrub import TableVectorizer

vectorizer = TableVectorizer()
vectorized_X = vectorizer.fit_transform(X)

HistGradientBoostingRegressor().fit(vectorized_X, y)
```

## 3. Simple Pipeline for tabular data
The TableVectorizer outputs data that can be understood by a scikit-learn estimator. Therefore we can easily build a 2-step scikit-learn Pipeline that we can fit, test or cross-validate and that works well on tabular data.
```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from skrub import TableVectorizer

pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())

results = cross_validate(pipeline, X, y)
scores = results["test_score"]
print(f"R2 score:  mean: {np.mean(scores):.3f}; std: {np.std(scores):.3f}")
print(f"mean fit time: {np.mean(results['fit_time']):.3f} seconds")
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
