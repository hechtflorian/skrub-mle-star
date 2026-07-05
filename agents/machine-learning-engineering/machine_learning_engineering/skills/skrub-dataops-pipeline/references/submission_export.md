# Submission export (submission agent only)

Load this **only** in the submission agent when appending test prediction and `./final/submission.csv` export. Earlier stages (init, ablation, tuning, ensemble) should stop after the holdout print.

## Block 1 vs Block 2

| Block | Bind | Purpose |
|---|---|---|
| **Block 1** (input — do not edit) | `skrub.var("data", train_part)` | Holdout metric → `print(f"Final Validation Performance: ...")` |
| **Block 2** (Your Task) | `skrub.var("data", train_df)` | Full-train refit → predict `test_df` → `./final/submission.csv` export |

**Reuse Block 1 verbatim** — same `fe`, `vectorizer`, `model`, legs, `get_proba`, blend/meta, thresholds, etc. — Only add Block 2 where you rebind `train_part` → full `train_df` and swap predict env from `valid_part` to `test_df` and then export `submission.csv`.

## Pattern 1 example — single estimator pipeline (append after holdout print)

Block 1 uses `data_train = skrub.var("data", train_part)` and ends with the holdout print. Append below that:

```python
test_df = pd.read_csv("./input/test.csv")   # load test_df if not present

data_full = skrub.var("data", train_df) # bind full train_df for refit
data_full = data_full.skb.apply_func(fe)  # same FE as Block 1
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)  # same nodes as Block 1
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)
```

## Pattern 2 example — ensemble / meta stack estimators (append after holdout print)

Block 1 builds each estimator leg on `train_part`, blends on validation, prints metric. Append — rebuild **every leg** to refit on full `train_df`, then the **same** blend/meta on test:

```python
test_df = pd.read_csv("./input/test.csv")

data_full = skrub.var("data", train_df) # load test_df if not present
data_full = data_full.skb.apply_func(fe)    # same FE as Block 1
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_a = X_full.skb.apply(vectorizer_a).skb.apply(model_a, y=y_full)   # example leg (same as Block 1)
full_pred_b = X_full.skb.apply(vectorizer_b).skb.apply(model_b, y=y_full)
learner_a = full_pred_a.skb.make_learner(fitted=True)
learner_b = full_pred_b.skb.make_learner(fitted=True)

test_p_a = get_proba(learner_a, test_df)  # same helper as Block 1
test_p_b = get_proba(learner_b, test_df)
test_blend = blend_or_meta(test_p_a, test_p_b)  # same function/weights/threshold as Block 1
test_pred = to_submission_labels(test_blend)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)
```

Replace example `blend_or_meta` / `to_submission_labels` with whatever Block 1 uses (weighted average, `meta_model.predict_proba`, threshold, etc.).

## Rules

- Append Block 2 **after** the holdout print.
- Reuse Block 1 pipeline names and graph; only change bind and predict target.
- Ensemble: keep all legs — do not merge to one model.
- Fit on `train_df`, predict on `test_df`; never concat train+test for fit (leakage).
- This reference shows generic examples — you must **reuse the exact pipeline you are given** to produce Block 2.

## Anti-patterns

- Re-pasting Block 1 or duplicating the whole script
- Collapsing multi-leg ensemble to a single model
- Adding a second `Final Validation Performance` print
- Rewriting the pipeline (FE, models, hyperparameters, blend logic, etc.)
