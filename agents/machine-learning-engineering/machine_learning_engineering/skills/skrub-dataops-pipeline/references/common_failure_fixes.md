# Common DataOps Failure Fixes

Use this during debugging. Keep DataOps architecture unchanged.

## 1) Chaining/Fit mismatch (`.fit` on DataOp output)
- Symptom: errors like `'Series' object has no attribute 'fit'` or internal `SkrubLearner` type errors.
- Fix:
  - Keep chain as DataOps and use `pred.skb.cross_validate()` for evaluation, or
  - compile first: `learner = pred.skb.make_learner(fitted=True)`.
  - If fitting manually after compile, fit with environment dict (not `X, y` positional form).

## 2) Predict environment violation
- Symptom: `TypeError: environment should be a dictionary of input values`.
- Fix:
  - Use `learner.predict({"data": test_df})` (or multiple named tables for multi-table flows).
  - Do not pass raw `DataOp` objects to `predict`.

## 3) Target or feature not tracked correctly
- Symptom: CV/search behaves incorrectly or raises shape/index issues.
- Fix:
  - Ensure features are marked with `skrub.X(...)` or `.skb.mark_as_X()`.
  - Ensure target is marked with `skrub.y(...)` or `.skb.mark_as_y()`.

## 4) DataOps pipeline accidentally downgraded to sklearn-only flow
- Symptom: code uses `TableVectorizer` but tuning/search is outside DataOps graph.
- Fix:
  - Rebuild main path as `.skb.apply(...)` chain.
  - Put tunables inside graph with `choose_*`/`choose_from`.
  - Run search from DataOp (`pred.skb.make_randomized_search(...)`).

## 5) Unsupported keyword arguments
- Symptom: errors from guessed kwargs like `TableVectorizer(impute_missing_values=True)` or `.skb.subsample(..., random_state=0)`.
- Fix:
  - Use only verified signatures from references in this skill.
  - For subsampling, use `.skb.subsample(n=...)` only.

## 5b) `choose_from` dict key type error
- Symptom: `TypeError` similar to "Outcome names should be of type str".
- Root cause: using non-string keys in dict-form `choose_from`.
- Fix:
  - use string keys: `skrub.choose_from({"6": 6, "8": 8}, name="max_depth")`
  - or replace with `skrub.choose_int(...)` for simple numeric ranges.

## 6) Multi-table merge mismatch
- Symptom: join/merge errors after aggregation.
- Fix:
  - Verify key columns before aggregation and merge.
  - Aggregate child-table features to one row per prediction unit before final merge.

## 7) Search errors or poor search behavior
- Symptom: search space empty or unexpected defaults.
- Fix:
  - Check `pred.skb.describe_param_grid()` first.
  - Confirm tunables use `choose_int`, `choose_float`, or `choose_from`.
  - For Optuna workflows, use `backend="optuna"` or `choose=trial`.

## 7b) "Fake tuning" (`choose_*` used but no search executed)
- Symptom: code defines `choose_*` / `choose_from(...)` but only calls `.skb.make_learner(...)` or `.skb.eval(...)` and still claims tuning.
- Root cause: `choose_*` defines a search space only; defaults are used unless randomized/grid/Optuna search is run.
- Fix:
  - For real tuning: run `.skb.make_randomized_search(...)` / `.skb.make_grid_search(...)` (or Optuna trial flow), then use best result (`search.best_learner_` or best trial learner).
  - For quick baseline: do not claim tuning; either keep default-choice behavior explicitly as baseline or replace `choose_*` with fixed constants for clarity.
  - Reuse prior best params when only minor non-hparam changes are made; avoid rerunning full search unless search-space-sensitive parts changed.

## 8) RMSE incompatibility in local sklearn version
- Symptom: `TypeError: got an unexpected keyword argument 'squared'` from `mean_squared_error(..., squared=False)`.
- Fix:
  - Compute RMSE as `mean_squared_error(y_true, y_pred) ** 0.5`.

## 9) Slow iteration during development
- Symptom: preview/build loop is too slow.
- Fix:
  - Use `.skb.subsample(n=...)` for previews.
  - Use `keep_subsampling=True` only for quick checks; run final CV without it.

## 10) Ablation agent stops after `list_skills` (no script output)
- Symptom: generated ablation file is empty or contains no runnable code.
- Root cause: tool calls were emitted, but final code response was not produced.
- Fix:
  - Treat tool calls as prep only.
  - Always end with executable ablation code.
  - Ensure ablation code prints per-variant metric and final best-variant summary.

## 11) Stuck at plain `TableVectorizer()` baseline
- Symptom: repeated runs only use `TableVectorizer()` defaults despite ablation indicating encoding/preprocessing impact.
- Root cause: plan focuses only on model hyperparameters and ignores encoding routing choices.
- Fix:
  - Load `references/encoding_skrub.md` and `references/selectors_routing_skrub.md` for routing choices.
  - Load `references/feature_engineering_skrub.md` if ablation should test ratios, drops, or cleaning.
  - Explicitly decide low/high-cardinality handling based on model family and data.
  - Keep the encoding changes in the DataOps path and validate with direct holdout RMSE.

## 12) Redundant target-column drop (`KeyError: '[target_col]' not found in axis`)
- Symptom: script crashes near prediction/inference with a pandas `KeyError` when doing something like `df.drop(columns=target_col)`.
- Root cause:
  - dropping `target_col` on a dataframe that already had it removed, or
  - dropping `target_col` on test data that never had a target column, or
  - applying multiple inconsistent train/test feature-prep branches.
- Fast fix:
  - Prefer one canonical feature split once: `X = data.drop(columns=target_col, errors="ignore")`.
  - For prediction, pass raw table environments to the learner (no extra ad-hoc drop): `learner.predict({"data": test_df})`.
  - If a drop is unavoidable in shared code, use `errors="ignore"` and keep the same helper for train/val/test.
- Prevention:
  - Do not re-drop target columns at inference time.
  - Keep train/validation/test feature preparation in one reusable function/path.

## 13) Refinement score equals baseline after adding derived features
- Symptom: RMSE unchanged after `.skb.apply_func(...)` adds new columns.
- Root cause: column routing used pre-FE pandas lists (`train_part.select_dtypes(...)`) so new
  columns never reach the model.
- Fix:
  - Use one `TableVectorizer()` on post-FE `X`, or route with `s.numeric()` / `s.string()` on `X`
    after `apply_func` (see `feature_engineering_skrub.md`).
- Prevention: do not build feature column lists from raw train data before FE transforms.

## 14) Ablation uses a different model than the solution under refinement (without explicit model-change hypothesis)
- Symptom: ablation RMSE (~56k HGB) does not match solution RMSE (~54k CatBoost); planner picks variants that regress.
- Root cause: ablation script swaps backbone model or split while testing feature blocks.
- Fix:
  - Reload `references/feature_engineering_skrub.md` and keep the same model family/params as `train_code`.
  - Ablate only the feature/preprocessing block; compare holdout RMSE on the same split.
- Prevention: ablation prompt requires same backbone unless model swap is the explicit hypothesis.

## When to load other references
- Load `dataops_api_quickmap.md` when rebuilding a broken DataOps path from a known-good template.
- Load `encoding_skrub.md` if failures are tied to weak/default encoding strategy.
- Load `selectors_routing_skrub.md` for split/concat routing or selector mistakes.
- Load `feature_engineering_skrub.md` for ratios, redundancy drops, cleaning, or ablation alignment.
- Load `choices_hparam_pattern.md` for `choose_*` semantics, fake-tuning prevention, or grid/randomized search fixes.
- Load `dataops_tuning_optuna.md` for Optuna-specific search/debug patterns.
- Load `joining_across_columns.md` for multi-table merge/aggregation correctness.