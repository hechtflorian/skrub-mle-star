---
name: skrub-dataops-pipeline
description: Build and debug end-to-end ML pipelines with skrub DataOps primitives, multi-table transformations, and choice-based tuning. Use this skill when generating or refining ML code that must stay DataOps-first.
---

# Skrub DataOps Pipeline
Use this skill to produce DataOps-native code (not sklearn-only orchestration with incidental skrub usage).
Load only the minimum references needed for the current step to avoid context pollution.

## Reference directory
- **DataOps core + minimum pipeline shape**: `references/dataops_api_quickmap.md`
  - General skrub DataOps usage. Always load if unsure about skrub DataOps usage.
- **DataOps choice-based hyperparameter tuning**: `references/choices_hparam_pattern.md`
  - Use when implementing or debugging `choose_*`/search logic. In refinement ablation, load when tuning is intentionally part of the current step.
- **Optuna integration for DataOps choices tuning**: `references/dataops_tuning_optuna.md`
  - Additional reference for advanced tuning; load when the current refinement step should use Optuna-backed search.
- **Vectorization and encoding strategy**: `references/encoding_skrub.md`
  - String/datetime encoders and `TableVectorizer`; includes encoder `choose_*` tuning (Pattern 2b).
- **Selectors and column routing**: `references/selectors_routing_skrub.md`
  - `ApplyToCols` / `DropCols` / selectors and DataOps split→concat routing.
- **Feature engineering and structural preprocessing**: `references/feature_engineering_skrub.md`
  - Derived features, redundancy/cleaning, scaling; TableReport-driven ablation patterns.
- **Multi-table DataOps pattern**: `references/joining_across_columns.md`
  - How to merge/group multiple tables.
- **General skrub API subset**: `references/skrub_general_api.md`
  - General skrub API usage beyond DataOps.
- **Fast iteration with preview subsampling**: `references/skrub_subsampling.md`
  - How to subsample with skrub.
- **Ablation study template (refinement)**: `references/ablation_dataops_template.md`
  - Minimal DataOps ablation shape, variant patterns, stdout contract. **Load for every ablation step.**
- **Tuning search template (terminal tuning)**: `references/tuning_dataops_template.md`
  - Minimal DataOps tune_implement shape; copy structural pipeline, inject `choose_*` on focus block only. **Load for every tune_implement step.**
- **Common skrub failure fixes**: `references/common_failure_fixes.md`
  - Known errors and quick fixes for skrub (DataOps) errors.
- **Holdout data leakage audit**: `references/holdout_data_leakage.md`
  - Incorrect vs correct bind/fit patterns; load for the data leakage checker agent.

## Rules (must follow)
1. Build the solution as a DataOps graph first, then write final code.
2. Keep the main training path in DataOps primitives (`skrub.var` or `skrub.X`/`skrub.y`, then `.skb.apply(...)`).
3. For tuning, place `skrub.choose_*` and `skrub.choose_from(...)` inside the DataOps graph, then call `.skb.make_randomized_search(...)` or `.skb.make_grid_search(...)` and use the best result.
4. `skrub.choose_*` without a search run is not tuning and should be used for fast DataOps graph validation only and never for final submissions: `.skb.make_learner(...)` and `.skb.eval(...)` on a graph with `choose_*` use default choice values only.
5. If compute budget does not allow search, do not leave `choose_*` placeholders in the final training path; replace them with explicit fixed values and state that this is a fixed-parameter run.
6. Never claim hyperparameter tuning unless search was actually executed and the final model comes from the search result (`best_learner_` or equivalent best-trial learner).
7. Keep DataOps architecture intact during debugging; patch uncertain calls, do not rewrite to sklearn-only pipelines.
8. If unsure, load the relevant reference and only use APIs present there.
9. Skill tool calls are preparation, not final output: After tool calls (`list_skills`/`load_skill`/`load_skill_resource`), you must always return the final output you are instructed to output (e.g., executable Python code, summary, plan).
10. If the step changes encoders or `TableVectorizer` config, load `references/encoding_skrub.md` before finalizing.
11. If the step changes column routing (`ApplyToCols`, `DropCols`, selectors, split/concat paths), load `references/selectors_routing_skrub.md`.
12. If the step adds derived features, drops redundant columns, cleans/scales numerics, or ablation profile suggests structural feature edits, load `references/feature_engineering_skrub.md`.
13. **Honest holdout validation:** `skrub.var("data", df)` defines what rows `make_learner(fitted=True)` trains on. For `Final Validation Performance`, bind **`train_part` only**, fit, then `predict({"data": valid_part})`. Do **not** load `test_df`, refit on full `train_df`, or write `submission.csv` in init, ablation, refinement, or tuning — **holdout metric only**. Full-train + test export is **submission stage only** (optional late ensemble export). See `references/dataops_api_quickmap.md`.

## Hard constraints (must follow to prevent common runtime failures)
- Do not call `.fit(X, y)` on a DataOp chain output.
- Do not pass DataOp nodes directly to `learner.predict(...)`; pass an environment dict, e.g. `{"data": df}`.
- Do not redundantly drop target columns during inference; avoid `test_df.drop(columns=target_col)` unless required (test_df usually doesnt include target_col), and if used, guard with `errors="ignore"`.
- Do not use unverified kwargs for `TableVectorizer(...)` or `.skb.subsample(...)`.
- Do not use `mean_squared_error(..., squared=False)` in this project runtime; for regression tasks that use `root_mean_squared_error`, compute RMSE as `mean_squared_error(y_true, y_pred) ** 0.5` instead.
- Silence training logs so stdout stays small: e.g. CatBoost `verbose=0`, LightGBM `verbose=-1`, XGBoost `verbosity=0` (never per-iteration logging).

## Default pipeline template (init, ablation, refinement, tuning)
Use the same `train_test_split` size and `random_state`. The final line should be **validation print** — no test load, no full-train refit if no submission export is expected.
```python
import numpy as np
import skrub
from sklearn.model_selection import train_test_split
# Use the competition metric from task_description.txt (# Metric section)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
predictor = X_train.skb.apply(vectorizer).skb.apply(YourModel(), y=y_train)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = your_metric_fn(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
```

**Submission stage only** (and optional final ensemble export): after the val metric line above, add full `train_df` refit + `test_df` predict. See `references/dataops_api_quickmap.md` → “Submission stage only”.

**Refinement ablation policy**:
- Load `references/ablation_dataops_template.md` and use its skeleton + print contract.
- Default to structural ablation with fixed/reused parameters.
- Escalate to bounded tuning only when the current hypothesis is explicitly about tuning impact. In this case, load `choices_hparam_pattern.md`.

**Refinement terminal tuning policy** (dedicated tuning stage after refinement):
- Load `references/tuning_dataops_template.md` and follow its skeleton for `tune_implement`.
- Structural ablation and plan/implement steps stay fixed-parameter.
- One bounded holdout randomized search on a **single** focus block (`model`, `encoder`, or `preprocessing`).
- Use `make_randomized_search` with `n_iter` from config; no CV, no Optuna.
- Search `n_jobs`: default 1 for multithreaded tree boosters (avoid nested parallelism / timeout surprises); 2 OK if estimator `n_jobs=1` or trials are cheap; 2 (max 4) for single-threaded sklearn.
- Bake best params into fixed code before ensemble/submission; downstream code must not contain `choose_*` or search calls.

## Output contract
- Return runnable single-file Python code when code is requested.
- Keep the metric print line with exact text: `Final Validation Performance: {final_validation_score}`.
