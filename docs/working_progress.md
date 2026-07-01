# Working progress (current state)

Rolling snapshot of recent MLE-STAR + skrub work. Full history: [`WORKING_PROGRESS_MLE_STAR_SKRUB.md`](./WORKING_PROGRESS_MLE_STAR_SKRUB.md) (see **Progress update 2026-07-01**).

**Scope:** `agents/machine-learning-engineering/` — OpenAI/ChatAI-compatible MLE-STAR with skrub DataOps, structural refinement, terminal tuning stage.

---

## Pipeline (current)

```
init → refinement (structural ablation → plan → implement)
     → [tuning: plan → search → bake → promote]   (config.tuning_enabled)
     → ensemble → submission
```

Downstream code after tuning must be **fixed-parameter** (no `choose_*`, no search).

---

## Shipped recently (2026-06-28 → 2026-07-01)

### Backbone drift guards

| Stage | Pre-exec mode | Anchor |
|-------|---------------|--------|
| `model_eval*` | **Strict** — exact estimator set | Retriever / first failing init script |
| `tune_implement*` | **Overlap** — ≥1 shared class with structural | `train_code_{outer_loop}_{task_id}` |
| `ablation*` | **Overlap** + `Ablation[` print contract | Input solution at refine step |
| `plan_implement*` | Prompt only | May swap backbone if plan says so |

- Unified API: `backbone_check_mode()`, `backbone_violation(..., mode=...)`, `canonical_estimator_set()` (LGBM/XGB aliases).
- **Tune overlap** (vs earlier strict equality) avoids false failures on ensemble one-leg tuning scripts.
- Reference: [`BACKBONE_DRIFT_GUARDS.md`](./BACKBONE_DRIFT_GUARDS.md).

### Tuning stage reliability

- **`map_tuning_best_params`:** value-range matching to plan specs (not `data_op__` index); fills from plan `default`; fixes swapped params in agent output.
- **Prompts/skills:** ensemble = tune one leg; human-named `TUNING_BEST_PARAMS`; CatBoost → Pattern 4 `choose_from`; `exec_timeout` in search budget text.
- **`common_failure_fixes.md` §1b:** `SkrubLearner` has no `.predict` when graph ends on transform instead of `.skb.apply(Estimator, y=y)`.
- **Tests:** `tests/test_code_util.py` — 22 passing.

### Search budget policy

| Knob | Where | Notes |
|------|--------|-------|
| `tuning_n_iter` | `config.py` | Max randomized search trials |
| `exec_timeout` | `config.py` → tune prompts | Wall-clock budget guidance |
| `tuning_n_jobs` | **Removed** | No longer in config or agent wiring |

**`n_jobs` (prompt-driven):** search parallelizes trials; tree boosters parallelize inside each fit.

- Default search **`n_jobs=1`** for LightGBM/CatBoost/XGBoost (predictable runtime — not because `n_jobs=2` fails).
- Search **`n_jobs=2`** OK if estimator **`n_jobs=1`** or trials are very cheap.
- Single-threaded sklearn: search **`n_jobs=2`** (max **`4`** if very fast). Never search **`n_jobs=-1`**.

**Grid search auto-pick** was prototyped then **reverted** — **randomized search only** for now.

### Manual tuning sandbox

```bash
cd agents/machine-learning-engineering/tests
uv run python test_tuning_script.py   # needs ./input/train.csv; not pytest
```

---

## Last validation run (Spaceship r8)

**Run:** `adk_run_20260701_160102` — artifacts under `test-runs/.../fix-refinement-debug-drift/r8/`

| Result | Notes |
|--------|-------|
| Best honest holdout ~**0.800** | LGBM init / refinement path |
| Tuning | Bad agent param mapping; **`tune_winner_source: structural`** (gate worked) |
| Guards | Init / ablation / refine drift checks behaved |
| **Open bug** | **Ensemble holdout leakage** (~0.978) — train/eval split mismatch across ensemble legs |

---

## Open items (priority order)

1. **Ensemble holdout leakage** — models fit on different splits, scored on shared `valid_part`.
2. Sync `BACKBONE_DRIFT_GUARDS.md` tune row with **overlap** mode if still strict.
3. Confirm tuning **wins** on fresh run after `map_tuning_best_params` fix (not just structural fallback).
4. Vanilla debug prompt / leakage checker patterns A–C (carried forward).

---

## Key files

```
agents/.../shared_libraries/config.py
agents/.../shared_libraries/code_util.py      # backbone + map_tuning_best_params
agents/.../sub_agents/tuning/{agent,prompt}.py
agents/.../skills/skrub-dataops-pipeline/references/
  tuning_dataops_template.md
  choices_hparam_pattern.md
  common_failure_fixes.md
agents/.../tests/test_code_util.py
agents/.../tests/test_tuning_script.py
```
