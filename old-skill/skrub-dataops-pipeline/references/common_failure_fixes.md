# Common DataOps Failure Fixes

Use this during debugging. Keep DataOps architecture unchanged.

## 1) Symbol not found / wrong import
- Verify symbol location with docs search.
- Update import path only; do not replace DataOps with sklearn pipeline code.

## 2) Wrong argument names in DataOps calls
- Patch only the uncertain call signature.
- Re-run quickly and keep all surrounding DataOps variables and apply-chain intact.

## 3) Shape/column mismatch after table transforms
- Trace where the mismatch appears in the DataOps graph.
- Fix the specific join/select/transform node causing mismatch.

## 4) Target/feature mix-up
- Ensure target is marked with `skrub.y` or `.skb.mark_as_y()`.
- Ensure features are marked with `skrub.X` or `.skb.mark_as_X()`.

## 5) Hidden sklearn-only fallback
- If the main training path is not on `.skb.apply(...)`, rewrite to DataOps-first orchestration.
- Keep any auxiliary utilities minimal and non-central.

## When to load other references
- Load `dataops_api_quickmap.md` when rebuilding a broken DataOps path from a known-good template.
- Load `encoding_skrub.md` if failures are tied to weak/default encoding or routing strategy.
- Load `choices_hparam_pattern.md` for `choose_*` semantics, fake-tuning prevention, or grid/randomized search fixes.
- Load `dataops_tuning_optuna.md` for Optuna-specific search/debug patterns.
- Load `joining_across_columns.md` for multi-table merge/aggregation correctness.