# DataOps Tuning with Optuna

Use this reference when the task requires Optuna-backed search for `choose_*` nodes in a DataOps plan.

## APIs quickmap
- `pred.skb.make_randomized_search(backend="optuna", ...)`
- `pred.skb.make_learner(choose=trial)`
- `skrub.cross_validate(learner, environment=env, cv=cv)`
- `search.study_` and `search.study_.best_params`

## Critical rule: no fake tuning with Optuna
- `choose_*` and `choose_from(...)` only become tuned hyperparameters when an Optuna search/study is actually executed.
- A plain `.skb.make_learner(...)` call on a graph containing `choose_*` still uses defaults and is not tuned; it should only serve as a quick baseline validation.
- Always produce the final model from the best search result (`search.best_learner_`) or best trial (`pred.skb.make_learner(choose=study.best_trial)`).

## Pattern A: Optuna as backend for DataOps randomized search
```python
import pandas as pd
import skrub
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

extra_tree = ExtraTreesRegressor(
    min_samples_leaf=skrub.choose_int(1, 32, log=True, name="min_samples_leaf")
)
ridge = Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="alpha"))
regressor = skrub.choose_from({"extra_tree": extra_tree, "ridge": ridge}, name="regressor")

data = skrub.var("data")
X = data.drop(columns="MedHouseVal", errors="ignore").skb.mark_as_X()
y = data["MedHouseVal"].skb.mark_as_y()
pred = X.skb.apply(regressor, y=y)

df = pd.read_csv(skrub.datasets.fetch_california_housing().path).sample(10_000, random_state=0)
env = {"data": df}
cv = KFold(n_splits=4, shuffle=True, random_state=0)

search = pred.skb.make_randomized_search(
    backend="optuna", cv=cv, n_iter=10, random_state=10
)
search.fit(env)
best_params = search.study_.best_params
```

## Pattern B: direct Optuna study with DataOps learner
```python
import optuna

def objective(trial):
    learner = pred.skb.make_learner(choose=trial)
    cv_results = skrub.cross_validate(learner, environment=env, cv=cv)
    return cv_results["test_score"].mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_learner = pred.skb.make_learner(choose=study.best_trial)
best_learner.fit(env)
```

## Checklist
- `choose_*`/`choose_from` exist before search.
- Optuna is used either through `backend="optuna"` or explicit `trial` workflow.
- Final model is created from best trial and fit on full environment.
- Holdout scripts (no CV): bind `train_part` for search/metric, `train_df` only for final test block — see `dataops_api_quickmap.md`.
- If no search is run, remove `choose_*` and switch to fixed values instead of leaving pseudo-tunable placeholders.

## When to load other references
- Load `dataops_api_quickmap.md` for canonical DataOps pipeline shape and safe fit/predict patterns.
- Load `choices_hparam_pattern.md` for non-Optuna choice/search conventions and tuning rules, and details on hyperparam search with `skrub.choose_*`
- Load `common_failure_fixes.md` for search-space/runtime failures and fast remediation.
- Load `encoding_skrub.md` when trials include encoder or column-routing decisions.
- Load `skrub_subsampling.md` when iteration speed is the bottleneck and subsampling is required.