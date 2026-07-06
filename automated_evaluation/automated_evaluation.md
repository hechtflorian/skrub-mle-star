# Automated Evaluation

`automated_evaluation/` provides a reproducible benchmark pipeline for MLE-STAR: define task metadata, execute agent runs, archive artifacts, and aggregate metrics into reports.

**Run all commands from the `mle-star_improved` repo root** (the directory that contains `automated_evaluation/` and `agents/`).

## Directory layout

```
automated_evaluation/
├── tasks_manifest.json           # Task metadata (generated from bundled task packs)
├── generate_tasks_manifest.py    # Step 0: build tasks_manifest.json
├── run_experiments.py            # Step 1: execute agent runs and archive artifacts
├── evaluate.py                   # Step 2: analyze archived runs and write reports
├── legacy/prepare_tasks.py       # Optional: download/split external datasets (legacy)
├── runs/<stamp>/                 # Raw run archives for each experiment batch
└── eval_results/<stamp>/         # Aggregated reports and JSON results
```

Task packs live under:

`agents/machine-learning-engineering/machine_learning_engineering/tasks/<task_name>/`

Each task folder needs `train.csv`, `test.csv`, and `task_description.txt`.

---

## One-time setup

### 1. Improved agent environment

```bash
cd agents/machine-learning-engineering
cp .env.example .env          # fill OPENAI_API_KEY, OPENAI_API_BASE, ROOT_AGENT_MODEL
uv sync
cd ../..                      # back to mle-star_improved root
```

The orchestrator launches agents via `uv run adk run` inside each agent checkout.

### 2. API configuration (`.env`)

Create `agents/machine-learning-engineering/.env` (see `.env.example`), at minimum:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE` (if using an OpenAI-compatible endpoint, e.g. ChatAI)
- `ROOT_AGENT_MODEL` (e.g. `openai/gpt-5.4-mini`)

### 3. Vanilla baseline (required when `--systems vanilla`)

Vanilla is a **sibling checkout** at `../mle-star_vanilla` (default). Bootstrap from the **improved repo root**:

```bash
./sh-scripts/bootstrap_vanilla_baseline.sh --revert-prompts
```

This creates a git worktree at `../mle-star_vanilla` pinned to the baseline commit with sklearn-only prompts (no skrub skills / TableReport / tuning stage).

Then set up vanilla the same way as improved:

```bash
cd ../mle-star_vanilla/agents/machine-learning-engineering
cp ../../mle-star_improved/agents/machine-learning-engineering/.env .   # or symlink
uv sync
cd ../../mle-star_improved    # back to improved root for experiments
```

**Custom vanilla path:** pass `--vanilla-agent-dir /path/to/vanilla/agents/machine-learning-engineering` to `run_experiments.py`.

**Re-bootstrap / fix prompts:** run `bootstrap_vanilla_baseline.sh` again with `--revert-prompts` (and optionally `--sync-tasks` to refresh task packs in the vanilla tree).

### 4. Task manifest

Generate before your first experiment batch (Step 0 below). Re-run after adding or renaming task folders.

---

## How to run

### Step 0: Generate task manifest

Scans bundled task folders and writes `tasks_manifest.json` (task type, metric, lower-is-better, target/id columns):

```bash
python automated_evaluation/generate_tasks_manifest.py
python automated_evaluation/generate_tasks_manifest.py --dry-run   # preview only
```

### Step 1: Execute experiments

```bash
# Preview the run matrix without executing agents
python automated_evaluation/run_experiments.py --dry-run

# Smoke test: one task, improved only
python automated_evaluation/run_experiments.py --tasks spaceship-titanic --systems improved

# Compare improved + vanilla on one task (sync task packs into vanilla checkout)
python automated_evaluation/run_experiments.py \
  --tasks spaceship-titanic \
  --systems improved vanilla \
  --sync-tasks-to-vanilla

# Full matrix: all tasks × improved + vanilla
python automated_evaluation/run_experiments.py --sync-tasks-to-vanilla

# Three repeats per (task, system)
python automated_evaluation/run_experiments.py --repeat-count 3 --sync-tasks-to-vanilla

# Resume: skip archives that already contain final_state.json
python automated_evaluation/run_experiments.py --skip-existing --sync-tasks-to-vanilla
```

**Improved vs vanilla:** one command runs both systems in sequence. Each system uses its own agent checkout and config flags (see table below). Use `--sync-tasks-to-vanilla` whenever vanilla is in the matrix so both checkouts see the same task packs.

**Improved only:** omit `--sync-tasks-to-vanilla` and pass `--systems improved`.

### Step 2: Analyze results

After Step 1 finishes, it prints the runs directory. Analyze with:

```bash
python automated_evaluation/evaluate.py --summarize-only \
  --runs-root automated_evaluation/runs/<stamp>
```

Reports land in `automated_evaluation/eval_results/<stamp>/`. Pointer: `eval_results/latest.json`.

---

## Environment policy (`uv sync`)

**Default: no `uv sync` between runs.** The agent checkout keeps a warm environment across tasks. If a run installs a package (e.g. `catboost`), it remains available for later tasks — faster and closer to typical iterative agent behavior.

Use `--uv-sync-before-run` when you need a clean dependency lock before each run (slow; resets to `pyproject.toml` / lockfile only).

By default, `run_experiments.py` streams the ADK log live to your terminal (same as `uv run adk run ...`) and prints `[i/total]` status lines before/after each cell. Pass `--quiet` to suppress live logs; if `tqdm` is installed, a progress bar is shown instead.

Document your chosen policy in experiment notes. For strict cross-task isolation, either enable `--uv-sync-before-run` or run one task per fresh checkout.

---

## Common CLI flags

| Flag | Script(s) | Description | Default |
|------|-----------|-------------|---------|
| `--tasks` | run, evaluate | Task subset (folder names) | All tasks under `tasks/` |
| `--systems` | run | `improved` / `vanilla` | Both |
| `--repeats` | run | Repeat labels, e.g. `run1 run2` | `run1` |
| `--repeat-count` | run | Generate `run1..runN` | — |
| `--seed` | run, evaluate | Random seed (patched into agent config) | `42` |
| `--model-label` | run, evaluate | Model name recorded in results | `ROOT_AGENT_MODEL` |
| `--runs-root` | run, evaluate | Root directory for run archives | `runs/<UTC stamp>` |
| `--improved-agent-dir` | run | Path to improved agent checkout | `agents/machine-learning-engineering` |
| `--vanilla-agent-dir` | run | Path to vanilla agent checkout | `../mle-star_vanilla/agents/...` |
| `--sync-tasks-to-vanilla` | run | Copy task packs into both agent checkouts | Off |
| `--uv-sync-before-run` | run | `uv sync` before each run | Off |
| `--skip-existing` | run | Skip completed archives | Off |
| `--dry-run` | run | Print matrix only | Off |
| `--quiet` | run | No live ADK log or status lines (tqdm if installed) | Off |
| `--summarize-only` | evaluate | Analyze existing runs only | Off |
| `--results-dir` | evaluate | Output directory for reports | `eval_results/` |

### Config patched per run

For each run, `shared_libraries/config.py` is temporarily patched with:

- `task_name`, `task_type`, `lower`, `seed`
- `use_data_leakage_checker=True`
- `table_report_enabled` / `tuning_enabled` per system (improved vs vanilla)

Original config is restored after each run.

### Systems under comparison

| Internal key | Archive folder | Agent checkout | Description |
|--------------|----------------|----------------|-------------|
| `improved` | `skrub-full` | `mle-star_improved/.../machine-learning-engineering` | TableReport + tuning + skrub skills enabled |
| `vanilla` | `vanilla` | `mle-star_vanilla/.../machine-learning-engineering` | Baseline: sklearn-only prompts; no TableReport / tuning / skills |

---

## Outputs

Each experiment batch writes under `automated_evaluation/runs/<stamp>/`:

```
runs/<stamp>/<task>/<system>/<model-slug>/<repeat>/
```

Example: `runs/20260705_210559/spaceship-titanic/skrub-full/gpt-5.4-mini/run1/`

Per-run archive contents:

| File / directory | Description |
|------------------|-------------|
| `final_state.json` | Final agent pipeline state (required for analysis) |
| `table_report.json` | Tabular data profile (improved only) |
| `meta.json` | Run metadata: task, metric, seed, config flags, timestamps |
| `analysis.json` | Per-run metrics from `test-scripts/analyze_run.py` |
| `1/`, `ensemble/` | Workspace stages |
| `adk_run_*.log` | ADK execution log |

Batch-level files:

| File | Description |
|------|-------------|
| `experiment_meta.json` | Batch config snapshot at start |
| `run_matrix.json` | Per-run orchestrator status |

Aggregated results (`evaluate.py --summarize-only`) go to `eval_results/<stamp>/`:

| File | Description |
|------|-------------|
| `report.md` | Human-readable report |
| `evaluation_summary.json` | Full structured bundle |
| `task_status.json` | Task readiness scan |
| `run_matrix.json` | Copy of execution matrix (if present) |

Pointer: `eval_results/latest.json`

---

## Benchmark tasks (current bundled set)

| Task | Type | Metric |
|------|------|--------|
| spaceship-titanic | Classification | accuracy |
| abalone-regression | Regression | RMSLE |
| blueberry-yield-regression | Regression | MAE |
| calories-burned-regression | Regression | RMSLE |
| covid19-forecasting-regression | Regression | RMSLE (multi-target) |
| podcast-listening-hours-regression | Regression | RMSE |
| diabetes-classification | Classification | ROC AUC |
| introverts-extroverts-classification | Classification | accuracy |
| bank-dataset-classification | Classification | ROC AUC |
| multi-class-pred-obesity-risk | Classification | accuracy |

---

## Example workflows

### Smoke test (improved only)

```bash
# From mle-star_improved root
python automated_evaluation/generate_tasks_manifest.py
python automated_evaluation/run_experiments.py --dry-run --tasks spaceship-titanic --systems improved
python automated_evaluation/run_experiments.py --tasks spaceship-titanic --systems improved
python automated_evaluation/evaluate.py --summarize-only \
  --runs-root automated_evaluation/runs/<stamp>
cat automated_evaluation/eval_results/latest.json
```

### Compare improved vs vanilla (one task)

```bash
# Prerequisites: bootstrap vanilla (see One-time setup) and .env in both checkouts
python automated_evaluation/generate_tasks_manifest.py
python automated_evaluation/run_experiments.py --dry-run \
  --tasks spaceship-titanic --systems improved vanilla --sync-tasks-to-vanilla
python automated_evaluation/run_experiments.py \
  --tasks spaceship-titanic --systems improved vanilla --sync-tasks-to-vanilla
python automated_evaluation/evaluate.py --summarize-only \
  --runs-root automated_evaluation/runs/<stamp>
```

A full 10-task × 2-system run can take a long time. Start with a single-task smoke test, then scale up. Use `--skip-existing` to resume long batches.
