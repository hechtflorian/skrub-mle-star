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
  - String/datetime encoders and `TableVectorizer` tuning.
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
- **Common skrub failure fixes**: `references/common_failure_fixes.md`
  - Known errors and quick fixes for skrub (DataOps) errors.

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

## Hard constraints (must follow to prevent common runtime failures)
- Do not call `.fit(X, y)` on a DataOp chain output.
- Do not pass DataOp nodes directly to `learner.predict(...)`; pass an environment dict, e.g. `{"data": df}`.
- Do not redundantly drop target columns during inference; avoid `test_df.drop(columns=target_col)` unless required (test_df usually doesnt include target_col), and if used, guard with `errors="ignore"`.
- Do not use unverified kwargs for `TableVectorizer(...)` or `.skb.subsample(...)`.
- Do not use `mean_squared_error(..., squared=False)` in this project runtime; compute RMSE as `mean_squared_error(...) ** 0.5`.

## Example skrub DataOps pipeline starter template
```python
import skrub

data = skrub.var("data", train_df)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = YourModel() # insert your model here
predictor = X_vec.skb.apply(model, y=y)

# Turning DataOps plan into fitted learner:
trained_learner = predictor.skb.make_learner(fitted=True)
preds = trained_learner.predict({"data": test_df})
```

Refinement ablation policy:
- Default to structural ablation with fixed/reused parameters.
- Escalate to bounded tuning only when the current hypothesis is explicitly about tuning impact. In this case, load `choices_hparam_pattern.md`.

Refinement terminal tuning policy (dedicated tuning stage after refinement):
- Structural ablation and plan/implement steps stay fixed-parameter.
- One bounded holdout randomized search on a **single** focus block (`model`, `encoder`, or `preprocessing`).
- Use `make_randomized_search` with low `n_iter` (default 4); no CV, no Optuna.
- Bake best params into fixed code before ensemble/submission; downstream code must not contain `choose_*` or search calls.

```python
# Example print contract inside ablation script
print(f"Ablation[{variant_name}] RMSE: {score}")
print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")
```

## Output contract
- Return runnable single-file Python code when code is requested.
- Keep the metric print line with exact text: `Final Validation Performance: {final_validation_score}`.
