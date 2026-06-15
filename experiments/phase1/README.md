# Phase 1 experiments

Vanilla MLE-STAR vs skrub-full on tabular tasks. See [`docs/experiment_plan_evaluation.md`](../../docs/experiment_plan_evaluation.md) for protocol.

## Layout

```
phase1/
  manifest.csv                 # 12-run index
  results/
    phase1_all_runs_12.csv     # primary aggregate (excl. legacy)
    phase1_all_runs.csv        # includes legacy skrub titanic run1
    phase1_by_task_system.csv  # mean ± std per (task, system)
    phase1_summary.md          # write-up
  california-housing-prices/
    vanilla/gpt-5.4-mini/run{1,2,3}/
    skrub-full/gpt-5.4-mini/run{1,2,3}/
  spaceship-titanic/
    vanilla/gpt-5.4-mini/run{1,2,3}/
    skrub-full/gpt-5.4-mini/run{1,2,3}/
    skrub-full/gpt-5.4-mini/legacy/run1/   # pre-grid; not in main stats
```

## Quick commands

```bash
# Per-run report
python test-scripts/analyze_run.py --path experiments/phase1/<task>/<system>/gpt-5.4-mini/run1

# Aggregate
python test-scripts/aggregate_runs.py \
  --root experiments/phase1 \
  --out experiments/phase1/results/phase1_all_runs.csv
```

## Config snapshot

| | California housing | Spaceship Titanic |
|--|-------------------|-------------------|
| `lower` | `True` (RMSE) | `False` (accuracy) |
| vanilla repo | `mle-star_vanilla` @ `eb59252` | same |
| skrub repo | `mle-star_improved` @ `b7e4adf` | same |
| seed | 42 | 42 |
