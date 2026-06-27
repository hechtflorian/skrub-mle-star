# Phase-1 experiment plan & evaluation guide

Concise checklist for **vanilla MLE-STAR vs skrub-enabled MLE-STAR** on two tabular tasks, with reliable artifacts and analysis.

**Run matrix:** 2 tasks × 2 systems × **2 repeats** = **8 runs**

| Task | Systems | Repeats |
|------|---------|---------|
| california-housing-prices (regression) | vanilla, skrub-full | run1, run2 |
| spaceship-titanic (classification) | vanilla, skrub-full | run1, run2 |

**Primary question:** does skrub-full beat vanilla on final holdout metric (mean over 2 repeats)?

**Secondary:** DataOps adherence, debug rounds, wall/exec time, tuning honesty (skrub only).

---

## Can multiple tasks live in `tasks/`?

**Yes.** Each task is a subfolder under `machine_learning_engineering/tasks/<task_name>/`. The pipeline loads whichever folder `config.task_name` points to (`prepare_task` reads `task_description.txt`; `create_workspace` copies that folder into `workspace/<task_name>/1/input/`).

You do **not** need separate code paths per task — only change `config.py` (and `lower` / `task_type` for classification) before each run.

**Recommended order:** complete the full workflow once on **housing only** (setup → run → archive → analyze), then add Spaceship Titanic. That isolates archive/eval bugs from task-packaging bugs.

---

## Directory layout (create once)

From repo root `mle-star_improved/`:

```
experiments/phase1/
  README.md                    # optional notes (model, dates, git shas)
  manifest.csv                 # one row per run — fill as you go
  california-housing-prices/
    vanilla/gpt-5.4-mini/run1/
    vanilla/gpt-5.4-mini/run2/
    skrub-full/gpt-5.4-mini/run1/
    skrub-full/gpt-5.4-mini/run2/
  spaceship-titanic/
    vanilla/.../run1|run2/
    skrub-full/.../run1|run2/
  eval/                        # Tier-B holdout labels (optional, see § Evaluation)
    california-housing-prices/holdout.csv
    spaceship-titanic/holdout.csv
  results/
    phase1_all_runs.csv        # from aggregate_runs.py
    phase1_summary.md          # your write-up
```

**Vanilla checkout** (sibling, not inside this repo):

```
~/uni/mle-star_vanilla/agents/machine-learning-engineering/   # clean MLE-STAR + infra fixes only
```

Record both git SHAs in every run’s `meta.json`.

---

## Phase 0 — One-time setup

### 0.1 Pin environment (both repos)

- [x] Same `ROOT_AGENT_MODEL` (e.g. `openai/gpt-5.4-mini`)
- [x] Same `.env` / ChatAI / LiteLLM settings
- [x] Same `exec_timeout`, `max_retry`, loop budgets in `config.py`
- [x] `seed = 42` for **all** runs (replication = run1/run2, not different seeds)

### 0.2 Vanilla baseline — bootstrap script + branch vs repo

**Recommendation: branch + git worktree in the same repo** (not a separate remote).

| Approach | Pros | Cons |
|----------|------|------|
| **Branch `vanilla-baseline` + worktree** (recommended) | Same history; `git diff vanilla-baseline..improve-refinement`; share `experiments/` + `--sync-tasks` | Don't merge skrub into that branch |
| Separate git repo | Hard isolation | Duplicate tasks/experiments, extra maintenance |

Baseline snapshot: **`ffa365c`** — ChatAI routing, DDG search, GPT-5 temperature, safe response parsing. **No** ADK skills, TableReport, or tuning. Prompts lightly mention skrub in prose (no skill toolset). For purer sklearn wording: `--revert-prompts`.

#### One command

From `mle-star_improved/` repo root:

```bash
./scripts/bootstrap_vanilla_baseline.sh --dest ../mle-star_vanilla
```

Useful flags:

```bash
# Sklearn-only prompts (6c96e03) + drop skrub/optuna deps
./scripts/bootstrap_vanilla_baseline.sh --dest ../mle-star_vanilla --revert-prompts

# Copy tasks/ from current HEAD (e.g. spaceship-titanic)
./scripts/bootstrap_vanilla_baseline.sh --dest ../mle-star_vanilla --sync-tasks

# Agents folder only, no git worktree
./scripts/bootstrap_vanilla_baseline.sh --mode export --dest ../mle-star_vanilla-export
```

#### After bootstrap

```bash
cd ../mle-star_vanilla/agents/machine-learning-engineering
cp ../../mle-star_improved/agents/machine-learning-engineering/.env .
uv sync
uv run adk run machine_learning_engineering
```

Compare vanilla vs skrub:

```bash
git diff vanilla-baseline..improve-refinement -- agents/machine-learning-engineering/machine_learning_engineering
```

Remove worktree when done: `git worktree remove ../mle-star_vanilla`

#### Checklist

- [x] `./scripts/bootstrap_vanilla_baseline.sh` completed (verification passed)
- [x] Same `.env` as skrub repo
- [ ] `--sync-tasks` once Spaceship Titanic exists on `improve-refinement`
- [x] Smoke housing → `final_state.json`
- [ ] Record SHA `ffa365c…` in `experiments/phase1/README.md`

### 0.3 Skrub repo (`mle-star_improved`)

- [x] Confirm defaults: `table_report_enabled=True`, `tuning_enabled=True`
- [x] Smoke: one housing run (your best recent config)

### 0.4 Analysis tools (this repo)

- [ ] `python test-scripts/analyze_run.py --path <archive_dir>`
- [ ] `python test-scripts/aggregate_runs.py --root experiments/phase1 --out experiments/phase1/results/phase1_all_runs.csv`

---

## Phase 1 — Housing only (validate workflow)

Complete this before adding Spaceship Titanic.

### 1.1 Configure (`mle-star_improved/.../config.py`)

```python
task_name = "california-housing-prices"
task_type = "Tabular Regression"
lower = True          # RMSE — lower is better
seed = 42
table_report_enabled = True   # skrub only; ignore in vanilla
tuning_enabled = True         # skrub only; vanilla has no tuning stage
```

### 1.2 Pre-run

- [ ] `cd agents/machine-learning-engineering && source .env` (or `set -a; source .env; set +a`)
- [ ] Confirm `task_name` matches folder under `tasks/`
- [ ] Optional: remove stale `workspace/california-housing-prices/` if you want a clean tree (pipeline recreates `1/` anyway)

### 1.3 Run

```bash
cd agents/machine-learning-engineering
uv run adk run machine_learning_engineering 2>&1 | tee run-logs/adk_run_$(date +%Y%m%d_%H%M%S).log
```

Repeat for **vanilla** repo with same env/model. Label runs `run1`, `run2` (same config both times).

### 1.4 Archive immediately (do not skip)

Copy from `workspace/california-housing-prices/` into  
`experiments/phase1/california-housing-prices/<system>/gpt-5.4-mini/runN/`:

| Artifact | Source |
|----------|--------|
| `final_state.json` | `workspace/<task>/final_state.json` |
| `adk_run_*.log` | `run-logs/` or workspace copy |
| `table_report.json` | `workspace/<task>/` (skrub only) |
| `1/` | all scripts (`train0.py`, `train1.py`, `ablation_0.py`, `train_tune_*.py`, …) |
| `ensemble/` | `ensemble0.py`, `ensemble1.py`, `final_solution.py`, `final/submission.csv` |
| `meta.json` | write by hand (see template below) |
| `analysis.json` | from analyze_run (next step) |

**`meta.json` template:**

```json
{
  "phase": "phase1",
  "task": "california-housing-prices",
  "system": "skrub-full",
  "repeat": "run1",
  "git_sha": "<short sha>",
  "repo": "mle-star_improved",
  "model": "openai/gpt-5.4-mini",
  "seed": 42,
  "table_report_enabled": true,
  "tuning_enabled": true,
  "started_at": "2026-06-13T...",
  "log_file": "adk_run_....log"
}
```

### 1.5 Per-run analysis

```bash
cd mle-star_improved
python test-scripts/analyze_run.py \
  --path experiments/phase1/california-housing-prices/skrub-full/gpt-5.4-mini/run1 \
  --json > experiments/phase1/.../run1/analysis.json
```

Human-readable report: same command without `--json`.

Append one row to `experiments/phase1/manifest.csv` (columns below).

### 1.6 Housing checklist

- [ ] vanilla run1 archived + analyzed
- [ ] vanilla run2 archived + analyzed
- [ ] skrub-full run1 archived + analyzed
- [ ] skrub-full run2 archived + analyzed

---

## Phase 2 — Add Spaceship Titanic

### 2.1 Download & prepare data

From [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data):

- `train.csv` — has label column **`Transported`** (boolean)
- `test.csv` — no `Transported` (same as housing: no public labels in agent `test.csv`)

Create:

```
agents/machine-learning-engineering/machine_learning_engineering/tasks/spaceship-titanic/
  train.csv
  test.csv
  task_description.txt
```

**Important:** use Kaggle’s `train.csv` / `test.csv` as-is for the agent (matches competition layout). Do **not** put solution labels in `test.csv`.

### 2.2 `task_description.txt` (minimal template)

Adapt column lists from your files if needed:

```markdown
# Task

Predict whether the passenger was transported to another dimension (Transported: True/False).

# Metric

accuracy

# Submission Format

PassengerId,Transported
0013_01,False
0018_01,False
etc.

# Dataset

train.csv — includes Transported
test.csv — features only, no Transported
```

### 2.3 `config.py` for Spaceship Titanic

```python
task_name = "spaceship-titanic"
task_type = "Tabular Classification"
lower = False         # accuracy — higher is better
seed = 42
```

Skrub flags same as housing. Re-run the same 4-run grid (2 vanilla + 2 skrub-full).

### 2.4 Spaceship-specific notes

- **Missing values** are common (`CryoSleep`, `Cabin`, `Age`, …) — expect TableVectorizer / encoders to matter; good stress test vs housing.
- **Metric line:** agents should print `Final Validation Performance: <accuracy>` on a holdout split; verify in log if scores look like 0–1 not RMSE-scale.
- **First skrub run:** skim `ablation_0.py` and `final_solution.py` for DataOps patterns and holdout-only validation (no test leakage).

### 2.5 Spaceship checklist

- [ ] Task folder created + description written
- [ ] One manual smoke run (skrub) before full grid
- [ ] vanilla run1 / run2 archived + analyzed
- [ ] skrub-full run1 / run2 archived + analyzed

---

## Phase 3 — Final evaluation

### Tier A — Internal holdout (start here)

Primary metric per run: **`score_submission`** from `analyze_run` / `to_row()` (holdout score printed by submission/final script — same protocol both systems use if prompts hold `random_state=42`).

```bash
python test-scripts/aggregate_runs.py \
  --root experiments/phase1 \
  --out experiments/phase1/results/phase1_all_runs.csv
```

**Compare per task:**

| Metric | How |
|--------|-----|
| Primary | mean ± std of `score_submission` over 2 runs; skrub vs vanilla delta |
| Win rate | skrub better in k/2 runs |
| Refinement gain | `gain_refinement` |
| Tuning gain | `gain_tuning` (skrub; often 0 if structural wins) |
| DataOps | `dataops_adherence` (skrub); vanilla should be ~0 on skrub checks |
| Debug cost | `debug_refinement`, `debug_tuning`, `contract_violations`, `context_errors` |
| Time | `wall_seconds`, `exec_seconds` |

**Sanity checks before trusting a delta:**

- [ ] `log_exit_code` is 0 (or document Ctrl-C aborts)
- [ ] No `context_errors` > 0
- [ ] `score_submission` not sentinel (`1e9`)
- [ ] Skrub tune scripts include structural FE (not bare TableVectorizer-only search)
- [ ] For classification, scores in [0, 1] not RMSE-scale

### Tier B — Fixed external holdout (optional, dont do this)

If Tier-A deltas are small or splits look inconsistent:

1. Split original Kaggle **train** once (e.g. 80/20, `random_state=42`).
2. Agent sees only 80% in `tasks/<task>/train.csv`.
3. Save 20% + labels in `experiments/phase1/eval/<task>/holdout.csv` (never copied to `tasks/`).
4. Add `test-scripts/score_holdout.py` later to score `final_solution.py` on that file.

Use Tier-B score as primary in write-up if you implement it; until then, Tier-A is acceptable for phase 1.

### Kaggle submission?

**Not required** for phase 1. Public/private leaderboard needs account + rate limits; your `test.csv` has no labels anyway. Revisit only if you want competition-calibrated numbers.

---

## `manifest.csv` columns

Maintain one row per archived run:

```csv
task,system,repeat,repo,git_sha,model,seed,table_report_enabled,tuning_enabled,archive_path,score_submission,gain_total,debug_tuning,wall_seconds,dataops_adherence,notes
```

Fill `score_*` and metrics from `analysis.json` → `row` object.

---

## Master todo (8 runs)

### Setup
- [ ] Phase 0 complete (env pinned, vanilla cloned, both smokes pass)
- [ ] `experiments/phase1/` tree created
- [ ] `manifest.csv` header written

### California housing (4 runs)
- [ ] vanilla run1 — run → archive → analyze → manifest row
- [ ] vanilla run2 — run → archive → analyze → manifest row
- [ ] skrub-full run1 — run → archive → analyze → manifest row
- [ ] skrub-full run2 — run → archive → analyze → manifest row

### Spaceship Titanic (4 runs)
- [ ] Task pack under `tasks/spaceship-titanic/`
- [ ] config: `task_name`, `task_type`, `lower=False`
- [ ] vanilla run1 / run2
- [ ] skrub-full run1 / run2

### Evaluation
- [ ] `aggregate_runs.py` → `results/phase1_all_runs.csv`
- [ ] Per-task summary: mean ± std, win rate, secondary metrics
- [ ] `results/phase1_summary.md` — short conclusion + known caveats

---

## Expansion (after phase 1)

1. Novelty 2×2 within skrub (`table_report_enabled` × `tuning_enabled`) — 4 configs × 2 tasks × 2 runs.
2. Tier-B holdout scorer + `archive_run.py` automation.
3. More tasks from MLE-bench lite using same `tasks/<name>/` layout.
4. Kaggle submit only for leaderboard alignment.

---

## Quick reference — config per run

| Field | Housing | Spaceship Titanic |
|-------|---------|-------------------|
| `task_name` | `california-housing-prices` | `spaceship-titanic` |
| `task_type` | Tabular Regression | Tabular Classification |
| `lower` | `True` | `False` |
| `seed` | `42` | `42` |
| `table_report_enabled` | skrub: `True` | skrub: `True` |
| `tuning_enabled` | skrub: `True` | skrub: `True` |

Change `task_name` (and `lower` / `task_type`) in `config.py`, run, archive — no other code changes required.
