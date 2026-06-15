# Holdout Data Leakage (Checker Reference)

Use when auditing skrub DataOps scripts for validation/test leakage.
Focus on the code that computes `Final Validation Performance` and any fit/predict path before that print.

## Core rule
`skrub.var("data", df)` and `.mark_as_X` / `skrub.X` defines which rows `make_learner(fitted=True)` trains on.
The learner must **not** see validation (or test) rows before the validation metric is printed.

## What to check
1. **Train scope before validation print:** model/preprocessing fit uses only `train_part` (or equivalent train fold), not full `train_df`.
2. **Validation eval:** `predict({"data": valid_part})` uses a learner fit on train rows only.
3. **Preprocessing fit scope:** transformers / `.skb.apply_func` stats (median, mean, encoders) must not learn from validation/test rows at metric time — bind `train_part` in Block 1.
4. **Full train timing:** `skrub.var("data", train_df)` and full-data `make_learner(fitted=True)` belong in **submission** — **after** printing `Final Validation Performance`, not in init/refinement/tuning scripts, i.e. use only if submission is expected.
5. **Tuning (if present):** `search.fit({"data": train_part})` and `search.best_learner_.predict({"data": valid_part})`; not `search.fit({"data": full_train_df})` for holdout scoring.
6. **Data splitting:** The `train_df` is split correctly into a `train_part` and `val_part` before further preprocessing, transforming, model fit etc. 

## Leakage signals (flag as leakage)

| Pattern | Why it leaks |
|---|---|
| `skrub.var("data", full_train_df)` → split → `make_learner(fitted=True)` → `predict({"data": valid_part})` for metric | Fit includes validation rows |
| Reusing a learner already fit on full `full_train_df` for holdout validation score | Same as above |
| `search.fit({"data": full_train_df})` then holdout eval on `valid_part` | Search trained on validation rows |
| Global pandas stats on full `full_train_df` used to transform `train_part` before metric (e.g. `train_df[col].median()` in shared prep) | Validation rows influenced preprocessing |
| Fitting encoders/scalers on concatenated train+val before split eval | Validation rows in fit |

## Correct patterns

**Block 1 — validation metric (fixed params):**
```python
data_train = skrub.var("data", train_part)
# ... same pipeline on X_train / y_train ...
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
print(f"Final Validation Performance: ...")
```

**Block 2 — test/submission stage only (after metric print):**
```python
data_full = skrub.var("data", train_df)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
```

Do **not** flag Block 2 as leakage when it appears only in submission-stage scripts.

**Tuning holdout:**
```python
search.fit({"data": train_part})
valid_pred = search.best_learner_.predict({"data": valid_part})
```

Rebuild the graph per block; do not score holdout with a learner fit on full data.

## Not leakage (do not flag)
- Full `train_df` refit + `test_df` predict in **submission-stage** code only (after validation print).
- Honest Block 1 with `train_part` binding in early-stage scripts (no test load required there).

## Extract / report contract
- Return the **smallest contiguous code block** that causes leakage (bind + fit + predict/eval path).
- If multiple issues exist, one entry per distinct leaky block is fine.

## Fix contract (refine step)
- Replace leaky bind/fit with Block 1 (`train_part`) for the metric path.
- Keep split constants (`test_size`, `random_state`) unchanged.
- Add Block 2 on `train_df` only after the validation print if test/submission code exists.
- Preserve DataOps structure, model, and pipeline logic; patch bind/fit scope only.
