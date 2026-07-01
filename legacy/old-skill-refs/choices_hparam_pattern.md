# Skrub Choices and Hyperparameter Pattern

High-level reference for building declarative tabular data preprocessing pipelines, relational joins, and DataOps graphs using the `skrub` library. `skrub` is a Python library built to seamlessly bridge tabular data (like pandas dataframes) to scikit-learn machine learning pipelines... modifiy this

Add: skrubs `choose_from` object allows to tune hyperparameters, choose optional configurations, and nest choices. Choices are not limited to choosing estimators and their hyperparameters: They can be used anywhere DataOps are used (e.g. argument of other DataOps' methods or operators).

## Agent Verification Checklist (Skrub)
Use this checklist when generating data preprocessing code:
- [ ] **No Pandas Engineering**: Did you use `skrub.TableVectorizer` to transform the dataframe to a vectorized representation?
- [ ] **DataOps Graph**: Did you define inputs using `skrub.X()`, `skrub.y()` or `skrub.var()` and `skrub.DataOp.skb.mark_as_X()`, `skrub.DataOp.skb.mark_as_y()`?
- [ ] **Relational Data**: If given multiple tables, did you use `skrub.Joiner` or `skrub.AggJoiner` instead of `pd.merge()`?

---

## 1. Basic example for hyperparameter tuning with skrub `choose_from` objects:
```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import skrub
import skrub.datasets

file_path = skrub.datasets.fetch_toxicity().path    # load example dataset
data = pd.read_csv(file_path)

# This dataset is sorted -- all toxic tweets appear first, so we shuffle it
data = data.sample(frac=1.0, random_state=1)

texts = data[["text"]]
labels = data["is_toxic"]
X = skrub.X(texts)
y = skrub.y(labels)

# Use skrub.choose_from() within DataOps plan to let skrub create scikit-learn hyperparameter-tuner (e.g. GridSearchCV) automatically
encoder = skrub.MinHashEncoder(
    n_components=skrub.choose_int(5, 15, n_steps=5, name="N components")
)

classifier = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)

# Use `pred` DataOp to perform hyperparam search with `.skb.make_grid_search()` or `skb.make_randomized_search()` -- they accept same arguments as their scikit-learn counterparts (e.g. `scoring`, `cv`, `n_jobs`)
search = pred.skb.make_randomized_search(
    n_iter=8, n_jobs=4, random_state=1, fitted=True
)
# search.results_   # show results

# Retrieve best learner with best hyperparam config found during search -- use to make predictions on new data
best_learner = serach.best_learner_
```

## 2. Choosing between multiple possible encoders with skrub `choose_from` objects:
```python
X, y = skrub.X(texts), skrub.y(labels)

n_components = skrub.choose_int(5, 15, name="N components")

encoder = skrub.choose_from(
    {
        "minhash": skrub.MinHashEncoder(n_components=n_components),
        "lse": skrub.StringEncoder(n_components=n_components),
    },
    name="encoder",
)
X.skb.apply(encoder, cols="text")
```

### 3. Choosing between multiple possible models with skrub `choose_from` objects:
```python
from sklearn.linear_model import RidgeClassifier

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
ridge = RidgeClassifier(alpha=skrub.choose_float(0.01, 100, log=True, name="α"))
classifier = skrub.choose_from({"hgb": hgb, "ridge": ridge}, name="classifier")
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
#print(pred.skb.describe_param_grid())

search = pred.skb.make_randomized_search(
    n_iter=16, n_jobs=4, random_state=1, fitted=True
)
#search.plot_results()
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
