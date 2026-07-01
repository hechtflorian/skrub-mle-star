# Tuning DataOps with Optuna

This example shows how to use Optuna to tune the hyperparameters of a skrub `DataOp`. Skrub DataOps contain "choices", objects created with `skrub.choose_from()`, `skrub.choose_int()`, `skrub.choose_float()`, etc. and we can use hyperparameter search techniques to pick the best outcome for each choice. Performing this search with Optuna allows us to benefit from its many features, such as state-of-the-art search strategies, monitoring and visualization, stopping and resuming searches, and parallel or distributed computation.

## Agent Verification Checklist (Skrub)
Use this checklist when generating Optuna-tuning code:
- [ ] **Choice nodes present**: Are tunables represented by `choose_*` / `choose_from(...)` in the DataOps graph?
- [ ] **Correct backend usage**: Is search created with `backend="optuna"` when using DataOps randomized search?
- [ ] **Trial integration**: If writing custom objective, is `choose=trial` passed to `make_learner(...)`?
- [ ] **Deployment path**: Is the best learner fitted on the full environment after tuning?

## Quick Reference
- **Optuna backend search**: `pred.skb.make_randomized_search(backend="optuna", ...)`
- **Custom trial learner**: `pred.skb.make_learner(choose=trial)`
- **Evaluate objective**: `skrub.cross_validate(learner, environment=env, cv=cv)`
- **Fit best learner**: `best_learner.fit(env)`

---

## 1. Simple regressor and example data
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

## When to load other references
- Load `dataops_api_quickmap.md` for canonical DataOps pipeline shape and safe fit/predict patterns.
- Load `choices_hparam_pattern.md` for non-Optuna choice/search conventions and tuning rules, and details on hyperparam search with `skrub.choose_*`
- Load `common_failure_fixes.md` for search-space/runtime failures and fast remediation.
- Load `encoding_skrub.md` when trials include encoder or column-routing decisions.
- Load `skrub_subsampling.md` when iteration speed is the bottleneck and subsampling is required.