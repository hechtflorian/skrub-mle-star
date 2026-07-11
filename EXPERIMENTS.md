# Experiments

How we ran the benchmark, where the artifacts live, and how to reproduce or extend the evaluation (including additional LLM models).

**Detailed harness reference:** [automated_evaluation/automated_evaluation.md](automated_evaluation/automated_evaluation.md)  
Experimental **config, metrics, and result analysis:** [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md)

All commands below assume the **improved checkout root** (`skrub-mle-star/`).

---

## Scripts to execute experiments


| Step | Script                                                                                             | Purpose                                                                                                                            |
| ---- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 0    | [automated_evaluation/generate_tasks_manifest.py](automated_evaluation/generate_tasks_manifest.py) | Build [tasks_manifest.json](automated_evaluation/tasks_manifest.json) from bundled task packs (already done, redo if tasks change) |
| 1    | [automated_evaluation/run_experiments.py](automated_evaluation/run_experiments.py)                 | Execute agent runs; archive by default under [runs/](automated_evaluation/runs/)                                                   |
| 2    | [automated_evaluation/evaluate.py](automated_evaluation/evaluate.py)                               | Analyze archived runs; write reports by default under [eval_results/](automated_evaluation/eval_results/)                          |
| —    | [test-scripts/analyze_run.py](test-scripts/analyze_run.py)                                         | Optional per-run deep dive (`final_state.json` + log)                                                                              |


### Exact commands (full pipeline)

```bash
# Step 0 — task manifest
python automated_evaluation/generate_tasks_manifest.py

# Step 1 — execute runs (writes to runs/<stamp>/)
python automated_evaluation/run_experiments.py \
  --runs-root automated_evaluation/runs/<stamp> \  # optional, will save as timestamp by default
  --systems improved  # or "vanilla" - optional, will run both if ommitted
  --tasks task1 task2 task3 ...  # optional, will execute all tasks when ommited
  --skip-existing  # optional, to make already existing results don't override/run again

# Step 2 — analyze (writes to eval_results/<stamp>/)
python automated_evaluation/evaluate.py --summarize-only \  # must use --summarize-only!
  --runs-root automated_evaluation/runs/<stamp>  # use same dir as saved by run_experiments.py
```

Before Step 1, load `.env` into your shell if you changed `ROOT_AGENT_MODEL` (see [One-time setup](#one-time-setup)).

---

## Result artifacts

### Primary batch: raw run archives

**Directory:** [automated_evaluation/runs/20260707_115040_full/](automated_evaluation/runs/20260707_115040_full/)


| File                  | Link                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------- |
| Batch config snapshot | [experiment_meta.json](automated_evaluation/runs/20260707_115040_full/experiment_meta.json) |


Layout per run:

```
runs/20260707_115040_full/<task>/<system>/<model-slug>/<repeat>/
```

Example archive:

[spaceship-titanic/skrub-full/gpt-5.4-mini/run1/](automated_evaluation/runs/20260707_115040_full/spaceship-titanic/skrub-full/gpt-5.4-mini/run1/)

Typical files inside each run folder: `final_state.json`, `adk_run_*.log`, `meta.json`, `analysis.json`, agent-produced scripts in `1/`, `ensemble/`. log, meta.json, analysis.json added by us, rest produced by mle-star.

### Primary batch — aggregated reports

**Directory:** [automated_evaluation/eval_results/20260707_115040_full/](automated_evaluation/eval_results/20260707_115040_full/)


| File                       | Link                                                                                                      |
| -------------------------- | --------------------------------------------------------------------------------------------------------- |
| Human-readable report      | [report.md](automated_evaluation/eval_results/20260707_115040_full/report.md)                             |
| Structured JSON bundle     | [evaluation_summary.json](automated_evaluation/eval_results/20260707_115040_full/evaluation_summary.json) |
| Task readiness scan        | [task_status.json](automated_evaluation/eval_results/20260707_115040_full/task_status.json)               |
| Pointer to latest analysis | [eval_results/latest.json](automated_evaluation/eval_results/latest.json)                                 |


Regenerate reports with Step 2 above (`evaluate.py --summarize-only`).

### Batch summary


| Property           | Value                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------- |
| Runs root          | [runs/20260707_115040_full/](automated_evaluation/runs/20260707_115040_full/)                 |
| Eval results       | [eval_results/20260707_115040_full/](automated_evaluation/eval_results/20260707_115040_full/) |
| Tasks              | 10 (see table below)                                                                          |
| Systems            | `skrub-full` (improved, ours) + `vanilla` (runtime compatibility applied)                     |
| Base models        | `openai/gpt-5.4-mini` , `openai/gpt-5.4`                                                      |
| Repeats            | `run1`                                                                                        |
| Seed               | `42`                                                                                          |
| Completed archives | 40 (`10 tasks × 2 systems x 2 base models`)                                                   |


### Tasks in this batch

10 tabular tasks, 5 classification, 5 regression.


| Task                                 | Type           | Metric                      |
| ------------------------------------ | -------------- | --------------------------- |
| spaceship-titanic                    | Classification | accuracy_score              |
| abalone-regression                   | Regression     | root_mean_squared_log_error |
| blueberry-yield-regression           | Regression     | mean_absolute_error         |
| bike-sharing-regression              | Regression     | root_mean_squared_log_error |
| covid19-forecasting-regression       | Regression     | root_mean_squared_log_error |
| employee-attrition-classification    | Classification | roc_auc_score               |
| introverts-extroverts-classification | Classification | accuracy_score              |
| multi-class-pred-obesity-risk        | Classification | accuracy_score              |
| reservation-cancel-classification    | Classification | roc_auc_score               |
| restaurant-revenue-regression        | Regression     | root_mean_squared_error     |


---

## One-time setup

See [automated_evaluation/automated_evaluation.md](automated_evaluation/automated_evaluation.md) for the full checklist. Minimum:

```bash
# Improved agent
cd agents/machine-learning-engineering
cp .env.example .env    # setup OPENAI_API_KEY, OPENAI_API_BASE, ROOT_AGENT_MODEL
uv sync
cd ../..

# Vanilla worktree (required for --systems vanilla)
git worktree add ../mle-star_vanilla vanilla-baseline   # skip if already present
cp agents/machine-learning-engineering/.env \
   ../mle-star_vanilla/agents/machine-learning-engineering/.env
(cd ../mle-star_vanilla/agents/machine-learning-engineering && uv sync)

# Task manifest
python automated_evaluation/generate_tasks_manifest.py
```

**Model selection:** set `ROOT_AGENT_MODEL` in both agent `.env` files (e.g. `openai/gpt-5.4-mini`).

**Load `.env` into your shell before [run_experiments.py](automated_evaluation/run_experiments.py).** The agent subprocess reads each checkout's `.env` automatically, but the orchestrator picks up `--model-label` (archive folder names, `experiment_meta.json`) from `**ROOT_AGENT_MODEL` in the current shell, not by re-reading `.env`. After editing `.env`, run:

```bash
# From improved agent dir (repeat for vanilla if running --systems vanilla)
cd agents/machine-learning-engineering
set -a
source .env
set +a
echo "$ROOT_AGENT_MODEL"   # sanity check
cd ../..                   # back to skrub-mle-star root
```

Then start experiments in **that same terminal**. A new terminal needs `source .env` again. Alternative: pass `--model-label openai/your-model` explicitly (no shell export needed).

**Agent behaviour** (tuning, TableReport, loop counts, etc.) comes from each checkout's `shared_libraries/config.py` — not overridden by the harness except `task_name`, `task_type`, `lower`, `seed`.

---

## How we executed this batch

The full matrix was **not** started as a single long job. We used a fixed runs root and resumed in chunks for resource risk management:

```bash
RUNS=automated_evaluation/runs/20260707_115040_full

# Example: first improved tasks
python automated_evaluation/run_experiments.py \
  --runs-root "$RUNS" \
  --tasks task1 task2 task3 \
  --systems improved \
  --skip-existing

# Then same for vanilla
python automated_evaluation/run_experiments.py \
  --runs-root "$RUNS" \
  --tasks task1 task2 task3 \
  --systems vanilla \
  --skip-existing

# Later: next batch of improved tasks, then again vanilla counterparts, etc. until done
```

`--skip-existing` skips archives that already have `final_state.json` **and** an ADK log ending at `[user]:`. Safe to re-run the same command after interruptions.

To reproduce the full matrix in one shot (long-running):

```bash
python automated_evaluation/run_experiments.py \
  --runs-root automated_evaluation/runs/<new-stamp>   # optional; automatically saved in runs/ with timestamp
```

That runs **all tasks × improved + vanilla** with seed `42`. Preview first:

```bash
python automated_evaluation/run_experiments.py --dry-run
```

Or run only one trial task live first (recommended):

```bash
python automated_evaluation/run_experiments.py \
  --tasks task1
  --system improved
```

Or run natively from `/agents/machine-learning-engineering` (not part of experiment, just system test without scripted experiment - results save to `/workspace` by default):

```bash
# Option 1: CLI
uv run adk run machine_learning_engineering

# Option 2: Web API
uv run adk web  # then select `machine_learning_engineering` from dropdown
```

---

## Analyze results automatically

Uses [evaluate.py](automated_evaluation/evaluate.py) (Must use `--summarize-only` flag! No agent execution, intended to be seperate via `run_experiments.py`).

### Batch report (vanilla vs skrub-full, per model)

```bash
python automated_evaluation/evaluate.py --summarize-only \
  --runs-root automated_evaluation/runs/<your-run>
```

Read:

- [eval_results/20260707_115040_full/report.md](automated_evaluation/eval_results/20260707_115040_full/report.md)

The report groups comparisons **per model** when multiple model folders exist under the same runs root. Simplest if all runs are collected in the same folder.

### Single-run inspection (optional)

Uses [analyze_run.py](test-scripts/analyze_run.py):

```bash
python test-scripts/analyze_run.py \
  --state automated_evaluation/runs/20260707_115040_full/spaceship-titanic/skrub-full/gpt-5.4-mini/run1/final_state.json \
  --workspace automated_evaluation/runs/20260707_115040_full/spaceship-titanic/skrub-full/gpt-5.4-mini/run1
```

Each archive also contains [analysis.json](automated_evaluation/runs/20260707_115040_full/spaceship-titanic/skrub-full/gpt-5.4-mini/run1/analysis.json) (written at archive time).

---

## Adding another model (e.g. Mistral)

You can append runs into the **same** runs root. Archives are keyed by model slug:

```
<task>/<system>/<model-slug>/<repeat>/
```

`gpt-5.4-mini` and `mistral-large-3-675b-instruct-2512` are **different paths** — existing GPT runs are not touched.

### Safe workflow

```bash
# 1. Point both agent .env files to the new model
#    ROOT_AGENT_MODEL=openai/mistral-large-3-675b-instruct-2512

# 2. Load into shell so archive paths use the new model slug
cd agents/machine-learning-engineering
set -a && source .env && set +a
echo "$ROOT_AGENT_MODEL"
cd ../..

# 3. Run into the same batch directory (for easier analysis later on)
python automated_evaluation/run_experiments.py \
  --runs-root automated_evaluation/runs/<your-run>\
  --skip-existing

# 4. Re-analyze everything (both models appear in report.md)
python automated_evaluation/evaluate.py --summarize-only \
  --runs-root automated_evaluation/runs/<your-run>
```

Or skip shell export and pass `--model-label openai/mistral-large-3-675b-instruct-2512` on the [run_experiments.py](automated_evaluation/run_experiments.py) line.

### What can overwrite?


| Action                                                                   | Effect                                                                                            |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| New model, same tasks                                                    | **Safe** — writes parallel folders under a new `<model-slug>/`                                    |
| Re-run same task + system + model + repeat **without** `--skip-existing` | **Overwrites** that one archive directory                                                         |
| Re-run with `--skip-existing`                                            | **Skips** completed archives; only runs missing/failed cells                                      |
| Each `run_experiments.py` invocation                                     | Overwrites batch-level `experiment_meta.json` and `run_matrix.json` (per-run archives unaffected) |
| `evaluate.py --summarize-only`                                           | Overwrites `eval_results/<runs-dir-name>/` for that runs root                                     |


**Recommendation:** keep `--skip-existing` when extending a batch for safety.

Optional: use a separate runs root per model (e.g. `runs/20260709_mistral_full/`) for cleaner provenance — analysis accepts any runs root.

---

## Adding another repeat (e.g. `run2`)

Same idea as a new model: the repeat label is part of the archive path, so `**run2` does not overwrite `run1`**.

```
.../spaceship-titanic/skrub-full/gpt-5.4-mini/run1/
.../spaceship-titanic/skrub-full/gpt-5.4-mini/run2/
```

Only re-running the **same** `(task, system, model, repeat)` overwrites that folder (e.g. `run1` again without `--skip-existing`).

### Add `run2` only (keep existing `run1`)

```bash
python automated_evaluation/run_experiments.py \
  --runs-root automated_evaluation/runs/20260707_115040_full \
  --repeats run2 \
  --skip-existing
```

Then re-analyze with [evaluate.py](automated_evaluation/evaluate.py) (see [Analyze results](#analyze-results-automatically)).

The report picks up **all** repeats under the runs root: per-run tables list `run1` and `run2`; summary tables show **mean ± std** across repeats.

### Notes

- Use `--repeats run2` when you only want the second run. `--repeat-count 2` schedules `run1` and `run2`; with `--skip-existing`, existing `run1` cells are skipped and only `run2` runs.
- Default seed is `42` for every repeat — runs are comparable but not independent random draws. Pass a different `--seed` (e.g. `43`) for `run2` if you want variation; record it in your notes.
- You can combine extensions: same runs root can hold multiple models **and** multiple repeats.

---

## Quick reference

```bash
# Preview matrix
python automated_evaluation/run_experiments.py --dry-run

# Smoke test (1 task live)
python automated_evaluation/run_experiments.py \
  --tasks spaceship-titanic --systems improved

# Execute multiple tasks but not all (recommended) - repeat for remaining tasks/system/base model setup and save in same dir until done
python automated_evaluation/run_experiments.py \
  --runs-root automated_evaluation/runs/<your_dir> --tasks task1 task2 task3 --systems improved

# Full benchmark (improved + vanilla)
python automated_evaluation/run_experiments.py

# Resume / extend (same model, new tasks or failed cells)
python automated_evaluation/run_experiments.py \
  --runs-root automated_evaluation/runs/20260707_115040_full \
  --skip-existing

# Add run2 only (keeps run1)
python automated_evaluation/run_experiments.py \
  --runs-root automated_evaluation/runs/20260707_115040_full \
  --repeats run2 \
  --skip-existing

# Analyze
python automated_evaluation/evaluate.py --summarize-only \
  --runs-root automated_evaluation/runs/20260707_115040_full
```

Summarized reports and analysis in [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md)