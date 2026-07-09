# Automated Evaluation

`automated_evaluation/` provides a reproducible benchmark pipeline for MLE-STAR: define task metadata, execute agent runs, archive artifacts, and aggregate metrics into reports.

**Run all commands from the improved checkout root** — the directory that contains `automated_evaluation/` and `agents/` (conventionally named `mle-star_improved/`).

## Repository layout

Improved and vanilla are **two branches of the same Git repository**, checked out as **sibling worktrees**:

```
<parent>/
├── mle-star_improved/          # branch: main — skrub-full (TableReport, tuning, skills)
│   ├── automated_evaluation/
│   └── agents/machine-learning-engineering/
└── mle-star_vanilla/           # branch: vanilla-baseline — baseline MLE-STAR (no skrub)
    └── agents/machine-learning-engineering/
```

| Branch | Worktree (default path) | Role |
|--------|-------------------------|------|
| `main` | `mle-star_improved/` | Improved agent (`--systems improved` → archive folder `skrub-full`) |
| `vanilla-baseline` | `../mle-star_vanilla/` | Vanilla baseline (`--systems vanilla` → archive folder `vanilla`) |

Both branches include the same bundled benchmark task packs under `machine_learning_engineering/tasks/`.

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

### 0. Clone and check out both branches

```bash
git clone https://git.tu-berlin.de/wang.zk25/mle-star_improved.git mle-star_improved
cd mle-star_improved
git checkout main

# Sibling worktree for vanilla (same repo, branch vanilla-baseline)
git worktree add ../mle-star_vanilla vanilla-baseline
```

If `../mle-star_vanilla` already exists, skip `git worktree add`.

**Custom layout:** pass `--vanilla-agent-dir /path/to/vanilla/agents/machine-learning-engineering` to `run_experiments.py` (default expects the sibling path above).

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

**Reload after editing `.env`.** Agent runs load each checkout's `.env` when `adk run` starts. The orchestrator's default `--model-label` (archive paths, `experiment_meta.json`) comes from **`ROOT_AGENT_MODEL` in your shell**, not from re-parsing `.env`. Before `run_experiments.py`, export it in the terminal you use for experiments:

```bash
cd agents/machine-learning-engineering
set -a
source .env
set +a
echo "$ROOT_AGENT_MODEL"   # verify
cd ../..                   # mle-star_improved root
```

Repeat in the vanilla agent dir if you run `--systems vanilla` (or copy/symlink the same `.env`). New terminals need `source .env` again. Alternatively, pass `--model-label openai/your-model` on every run.

### 3. Vanilla baseline (required when `--systems vanilla`)

The `vanilla-baseline` branch is maintained in the repo (standard (sklearn) prompts; no TableReport / tuning / skrub skills). After the worktree from step 0, set up the agent environment the same way as improved:

```bash
cd ../mle-star_vanilla/agents/machine-learning-engineering
cp ../../../mle-star_improved/agents/machine-learning-engineering/.env .   # or symlink
uv sync
cd ../../../mle-star_improved    # back to improved root for experiments
```

**Task packs:** both branches already ship the bundled tasks. Use `--sync-tasks-to-vanilla` only when you changed tasks on `main` and want to copy them into the vanilla worktree before a run.

**Legacy bootstrap:** `sh-scripts/bootstrap_vanilla_baseline.sh` recreates vanilla from an old commit in git history. Prefer the `vanilla-baseline` worktree above for normal evaluation; use the script only when reproducing a historical baseline snapshot. Make sure prompts do not mention skrub, or run `--revert-prompts`.

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

If you changed `ROOT_AGENT_MODEL` in `.env`, `source .env` in this terminal first (see **§2 Reload after editing `.env`**).

```bash
# Preview the run matrix without executing agents
python automated_evaluation/run_experiments.py --dry-run

# Smoke test: one task, improved only
python automated_evaluation/run_experiments.py --tasks spaceship-titanic --systems improved

# Compare improved + vanilla on one task (both worktrees from One-time setup)
python automated_evaluation/run_experiments.py \
  --tasks spaceship-titanic \
  --systems improved vanilla

# All tasks x one version
python automated_evaluation/run_experiments.py --systems improved

# Full matrix: all tasks × improved + vanilla
python automated_evaluation/run_experiments.py

# Three repeats per (task, system) - expensive
python automated_evaluation/run_experiments.py --repeat-count 3

# After editing task packs on main, sync into vanilla before comparing
python automated_evaluation/run_experiments.py --sync-tasks-to-vanilla

# Resume: skip archives that already completed successfully
python automated_evaluation/run_experiments.py --skip-existing
```

**Improved vs vanilla:** one command runs both systems in sequence. Each system uses its own worktree and config flags (see table below). Task sync is optional because both branches bundle the same tasks; use `--sync-tasks-to-vanilla` when `main` has newer task packs than `vanilla-baseline`.

**Improved only:** pass `--systems improved` (default still runs both — set explicitly to skip vanilla).

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
| `--model-label` | run, evaluate | Model name recorded in results and archive paths | Shell `ROOT_AGENT_MODEL` at script start, or pass explicitly (see §2 — not read from agent `.env`) |
| `--runs-root` | run, evaluate | Root directory for run archives | `runs/<UTC stamp>` |
| `--improved-agent-dir` | run | Path to improved agent checkout | `agents/machine-learning-engineering` |
| `--vanilla-agent-dir` | run | Path to vanilla agent checkout | `../mle-star_vanilla/agents/...` |
| `--sync-tasks-to-vanilla` | run | Copy task packs from improved into both agent checkouts | Off |
| `--uv-sync-before-run` | run | `uv sync` before each run | Off |
| `--skip-existing` | run | Skip archives with `final_state.json` and ADK prompt return in log | Off |
| `--dry-run` | run | Print matrix only | Off |
| `--quiet` | run | No live ADK log or status lines (tqdm if installed) | Off |
| `--summarize-only` | evaluate | Analyze existing runs only | Off |
| `--results-dir` | evaluate | Output directory for reports | `eval_results/` |

### Config patched per run

For each run, `shared_libraries/config.py` is temporarily patched with task-specific fields only:

- `task_name`, `task_type`, `lower`, `seed`
- `seed` (default=42) is patched only if given as CLI arg

All other agent settings (`use_data_leakage_checker`, `table_report_enabled`, `tuning_enabled`, loop counts, `num_solutions`, etc.) are read from each agent checkout's `config.py` and are not overridden by the orchestrator. Original config is restored after each run.

=> each agents `config.py` is the **single source of truth**. Make sure they align for comparability. 

### Systems under comparison

| Internal key | Archive folder | Worktree / branch | Description |
|--------------|----------------|-------------------|-------------|
| `improved` | `skrub-full` | `mle-star_improved/` · `main` | Skrub skills + agent defaults in improved `config.py` |
| `vanilla` | `vanilla` | `mle-star_vanilla/` · `vanilla-baseline` | Baseline prompts + agent defaults in vanilla `config.py` |

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
| `adk_run_*.log` | ADK execution log (second `[user]:` prompt at end ⇒ agent finished) |
| `table_report.json` | Tabular data profile (improved only) |
| `meta.json` | Run metadata: task, metric, seed, `agent_config_from_file`, timestamps; after run also `agent_config` from `final_state.json` |
| `analysis.json` | Per-run metrics from `test-scripts/analyze_run.py` |
| `1/`, `ensemble/` | Workspace stages |

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
# Prerequisites: clone + worktrees (see One-time setup) and .env in both agent dirs
python automated_evaluation/generate_tasks_manifest.py
python automated_evaluation/run_experiments.py --dry-run \
  --tasks spaceship-titanic --systems improved vanilla
python automated_evaluation/run_experiments.py \
  --tasks spaceship-titanic --systems improved vanilla
python automated_evaluation/evaluate.py --summarize-only \
  --runs-root automated_evaluation/runs/<stamp>
```

A full 10-task × 2-system run can take a long time. Start with a single-task smoke test, then scale up. Use `--skip-existing` to resume long batches.
