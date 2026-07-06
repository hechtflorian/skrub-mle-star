# Holdout Data Leakage (Checker Reference)

Use when auditing skrub DataOps scripts for **holdout metric leakage** (init, ablation, refinement, tuning, ensemble).

Focus on code **before** `print(f"Final Validation Performance: ...")`. Submission export is handled separately — not checked here.

## Core rule

`skrub.var("data", df)` defines which rows `make_learner(fitted=True)` trains on.

Before the holdout print, the learner and any preprocessing must **not** see validation rows — and must **not** use test rows (from `test.csv`) at all in early-stage scripts.

## What to check

1. **Train bind:** holdout path binds `train_part` (or train fold), not full `train_df`.
2. **Validation eval:** `predict({"data": valid_part})` uses a learner fit on train rows only.
3. **Preprocessing scope:** `.skb.apply_func`, encoders, imputers learn only from train rows bound before the metric.
4. **No test data:** no `test_df` / `test.csv` load, no `skrub.var("data", test_df)`, no train+test concat in early scripts.
5. **Tuning (if present):** `search.fit({"data": train_part})`, not `search.fit({"data": train_df})` for holdout scoring.
6. **Split order:** `train_test_split` → bind `train_part` → build pipeline → score `valid_part` → print metric.

## Leakage signals (flag as leakage)

| Pattern | Why it leaks |
|---|---|
| `skrub.var("data", train_df)` → `make_learner(fitted=True)` → `predict({"data": valid_part})` for metric | Fit includes validation rows |
| Split **after** binding or fitting on full `train_df` | Validation rows in fit |
| Reusing a learner fit on full `train_df` to score `valid_part` | Same as above |
| `search.fit({"data": train_df})` then eval on `valid_part` | Search saw validation rows |
| `test_df` / `test.csv` loaded or bound before holdout print | Test rows must not appear in early scripts |
| `skrub.var("data", test_df)` or train+test concat used in fit/preprocessing before metric | Test or val rows influence training |
| Global pandas stats on full `train_df` feeding transforms on `train_part` (e.g. `train_df[col].median()`) | Validation rows influenced preprocessing |
| Encoders/scalers fit on concatenated train+val before holdout eval | Validation rows in fit |

## Correct holdout pattern

```python
train_part, valid_part = train_test_split(train_df, ...)

data_train = skrub.var("data", train_part)
# ... pipeline on train bind only ...
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
print(f"Final Validation Performance: ...")
```

**Tuning holdout:**

```python
search.fit({"data": train_part})
valid_pred = search.best_learner_.predict({"data": valid_part})
```

Rebuild the graph on `train_part`; never score holdout with a learner fit on full `train_df`.

## Not leakage (do not flag)

- Honest `train_part` bind, `valid_part` predict, single holdout print.
- No `test_df`, no full-train refit, no `submission.csv` in early-stage scripts (expected — stop after metric).

## Extract / report contract

- Return the **smallest contiguous block** causing leakage (bind + fit + predict/eval path).
- One entry per distinct leaky block is fine.

## Fix contract (refine step)

- Rebind leaky path to `train_part` for the metric; keep `test_size`, `random_state` unchanged.
- Remove `test_df` load, full-train refit, or `submission.csv` from early-stage scripts if present.
- Preserve DataOps structure and model logic; patch bind/fit scope only.
- Return the **smallest patched block** for in-place replace — not the full file unless necessary.

## When to load other references
- Load `dataops_api_quickmap.md` for correct holdout bind patterns.
- Load `submission_export.md` for submission Block 2 (submission agent only).
- Load `common_failure_fixes.md` for leakage symptom fixes (#15).
