# Tuning DataOps with Optuna

This example shows how to use Optuna to tune the hyperparameters of a skrub `DataOp`. Skrub DataOps contain "choices", objects created with `skrub.choose_from()`, `skrub.choose_int()`, `skrub.choose_float()`, etc. and we can use hyperparameter search techniques to pick the best outcome for each choice. Performing this search with Optuna allows us to benefit from its many features, such as state-of-the-art search strategies, monitoring and visualization, stopping and resuming searches, and parallel or distributed computation.

## Agent Verification Checklist (Skrub) - MODIFY THIS
Use this checklist when generating data preprocessing code:
- [ ] **No Pandas Engineering**: Did you use `skrub.TableVectorizer` to transform the dataframe to a vectorized representation?
- [ ] **DataOps Graph**: Did you define inputs using `skrub.X()`, `skrub.y()` or `skrub.var()` and `skrub.DataOp.skb.mark_as_X()`, `skrub.DataOp.skb.mark_as_y()`?
- [ ] **Relational Data**: If given multiple tables, did you use `skrub.Joiner` or `skrub.AggJoiner` instead of `pd.merge()`?

---

## 1. Simple Regressor and example data
We will fit a regressor containing a few choices on a toy dataset. We try 2 regressors: extra trees and ridge. They both have hyperparameters that we want to tune.
```python
# pip install optuna
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

import optuna
import skrub

extra_tree = ExtraTreesRegressor(
    min_samples_leaf=skrub.choose_int(1, 32, log=True, name="min_samples_leaf"),
)
ridge = Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α"))

regressor = skrub.choose_from(
    {"extra_tree": extra_tree, "ridge": ridge}, name="regressor"
)
data = skrub.var("data")
X = data.drop(columns="MedHouseVal", errors="ignore").skb.mark_as_X()
y = data["MedHouseVal"].skb.mark_as_y()
pred = X.skb.apply(regressor, y=y)
#print(pred.skb.describe_param_grid())

# Load data
# (We subsample the dataset by half to make the example run faster)
file_path = skrub.datasets.fetch_california_housing().path
df = pd.read_csv(file_path).sample(10_000, random_state=0)

# The environment we will use to fit the learners created by our DataOp.
env = {"data": df}
cv = KFold(n_splits=4, shuffle=True, random_state=0)

# Selecting the best hyperparams with Optuna
search = pred.skb.make_randomized_search(
    backend="optuna", cv=cv, n_iter=10, random_state=10
)
search.fit(env)
#search.results_

# The Optuna Study that was used to run the hyperparameter search is available in the attribute `study_`
#search.study_
#search.study_.best_params

def objective(trial):
    learner = pred.skb.make_learner(choose=trial)
    cv_results = skrub.cross_validate(learner, environment=env, cv=cv)
    return cv_results["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
#study.best_params

# Build a learner with best hyperparams and fit on full dataset
best_learner = pred.skb.make_learner(choose=study.best_trial)
best_learner.fit(env)
#print(best_learner.describe_params())
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