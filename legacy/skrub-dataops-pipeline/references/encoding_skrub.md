# Encoding: from a dataframe to a numerical matrix for ML

This example shows how to transform a rich dataframe with columns of various types into a numerical matrix on which machine-learning algorithms can be applied. This example explores `skrub.TableVectorizer`: The TableVectorizer bridges the gap between tabular data and machine-learning pipelines. It allows us to apply a machine-learning estimator to our dataframe without manual data wrangling and feature extraction. 

The TableVectorizer distinguishes between 4 basic kinds of columns. For each kind, it applies a different transformation, which we can configure. The kinds of columns and the default transformation for each of them are:
- numeric columns: simply casting to floating-point
- datetime columns: extracting features such as year, day, hour with the `DatetimeEncoder`
- low-cardinality categorical columns: one-hot encoding
- high-cardinality categorical columns: a simple and effective text representation pipeline provided by the `GapEncoder`

## Agent Verification Checklist (Skrub)
Use this checklist when generating encoding/preprocessing code:
- [ ] **Vectorization first**: Is `TableVectorizer` or `tabular_pipeline` used instead of manual encoding chains where possible?
- [ ] **Column type handling**: Are numeric, datetime, low-cardinality, and high-cardinality columns treated with suitable encoders?
- [ ] **Pipeline compatibility**: Is the vectorized output fed into a supervised learner through a valid sklearn/skrub pipeline?
- [ ] **Task fit**: Is the chosen encoder strategy consistent with the dataset size/cardinality?

## Quick Reference
- **Fast baseline**: `tabular_pipeline("regression")` / `tabular_pipeline("classification")`
- **Direct vectorization**: `TableVectorizer().fit_transform(X)`
- **Pipeline pattern**: `make_pipeline(TableVectorizer(), Estimator())`
- **High-cardinality tuning**: `TableVectorizer(high_cardinality=...)`
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

## When to load other references
- Load `dataops_api_quickmap.md` if DataOps variable/learner/predict contracts need confirmation.
- Load `choices_hparam_pattern.md` when encoding options are being tuned via `choose_*` / `choose_from`.
- Load `dataops_tuning_optuna.md` when encoding/model selection is optimized with Optuna trials.
- Load `common_failure_fixes.md` for pipeline breakages, API misuse, or metric/contract issues.
- Load `skrub_general_api.md` for additional non-DataOps transformer/selector examples.