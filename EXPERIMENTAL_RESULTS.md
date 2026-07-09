# Experimental results

Detailed discussion of the benchmark batch and what it says about our modifications (skrub DataOps skill, TableReport profiling, tuning stage, drift/robustness guards). For **how** to run/reproduce, see [EXPERIMENTS.md](EXPERIMENTS.md); the raw report this write-up is based on is `[eval_results/20260707_115040_full/report.md](automated_evaluation/eval_results/20260707_115040_full/report.md)`.

## Setup at a glance


|         |                                                                                                                                                                                           |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Batch   | `[runs/20260707_115040_full/](automated_evaluation/runs/20260707_115040_full/)` · reports `[eval_results/20260707_115040_full/](automated_evaluation/eval_results/20260707_115040_full/)` |
| Systems | `skrub-full` (our improved pipeline) vs `vanilla` (upstream + our runtime-compat layer)                                                                                                   |
| Model   | `openai/gpt-5.4-mini`                                                                                                                                                                     |
| Tasks   | 10 (6 regression, 4 classification)                                                                                                                                                       |
| Repeats | `run1` only — **N = 1 per cell**                                                                                                                                                          |
| Seed    | `42`                                                                                                                                                                                      |
| Scores  | **holdout validation** parsed from `Final Validation Performance:` in each stage script — **not** Kaggle/test-set leaderboard scores                                                      |


Machine-readable versions of everything below: `[evaluation_summary.json](automated_evaluation/eval_results/20260707_115040_full/evaluation_summary.json)` · `[task_status.json](automated_evaluation/eval_results/20260707_115040_full/task_status.json)`.

### Agent configuration used (`config.py`)

Captured per system in [`experiment_meta.json → agent_configs`](automated_evaluation/runs/20260707_115040_full/experiment_meta.json). Both systems ran with identical shared knobs (small loop counts for a tractable 10-task × 2-system batch); the improved system additionally enabled TableReport + tuning. Per-task, the harness overrides only `task_name`, `task_type`, `lower`, and `seed`.

| `config.py` knob | Value (both systems) | Meaning |
|---|---|---|
| `num_solutions` | 1 | parallel solution legs |
| `num_model_candidates` | 2 | candidate models in init retrieval |
| `inner_loop_round` | 1 | refinement inner (plan-refine) rounds |
| `outer_loop_round` | 1 | refinement outer (ablation) rounds |
| `ensemble_loop_round` | 1 | ensemble refine rounds |
| `num_top_plans` | 2 | plans kept for refinement/ensemble |
| `max_debug_round` | 5 | debug retries per stage |
| `max_rollback_round` | 2 | rollbacks on repeated failure |
| `max_retry` | 10 | generic op retries |
| `exec_timeout` | 600 s | per-script execution cap |
| `seed` | 42 | reproducibility |
| `use_data_leakage_checker` | false | leakage-checker sub-agent (off) |
| `use_data_usage_checker` | false | data-usage checker (off) |

| Improved-only flag | Value | Meaning |
|---|---|---|
| `table_report_enabled` | true | TableReport profile injected into refinement planners ([§3/§11](CONTRIBUTIONS.md#3-tablereport-data-profiling-improved-only)) |
| `tuning_enabled` | true | terminal tuning stage active ([§12](CONTRIBUTIONS.md#12-new-tuning-stage-sub_agentstuning-improved-only)) |
| `tuning_n_iter` | 5 | max randomized-search iterations in tuning |

Model temperature was normalized to `1.0` for the GPT-5 family via `get_compatible_temperature` (LiteLLM compat, [§1](CONTRIBUTIONS.md#1-openai--chatai-runtime-compatibility-shared-with-vanilla)). Full defaults live in [`shared_libraries/config.py`](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/config.py).

> **Read scores carefully.** We report two holdout numbers per task (see the [metric guide](#metric-guide)): **(best)** = the upstream script the submission agent received (best structural/ensemble solution), **(sub)** = what the submission script itself printed. When the submission print looks anomalous (e.g. `0` or suspiciously better than upstream), **(best)** is the fair comparison. With N = 1, treat everything below as directional, not statistically significant.

## Headline: primary metric (accuracy / task metric)

On the fair **(best)** comparison, vanilla wins more tasks than skrub-full with this model:

- **Upstream (best):** skrub-full wins **3/10**, vanilla wins **7/10**; ties **0**, while 2 tasks' scores are almost similar.
- **Submission (sub):** skrub-full wins **2/10**, vanilla wins **5/10** (3 tasks had no parseable vanilla submission score (see caveats).

But the wins are lopsided in magnitude; **skrub-full's wins are large, its losses are mostly tiny:**


| Task                                 | Metric    | Vanilla (best) | Skrub-full (best) | Δ (skrub−vanilla, dir-aware) | Winner         |
| ------------------------------------ | --------- | -------------- | ----------------- | ---------------------------- | -------------- |
| bike-sharing-regression              | RMSLE ↓   | 0.2991         | **0.0144**        | **+0.2846**                  | skrub-full     |
| restaurant-revenue-regression        | RMSE ↓    | 3,201,219      | **2,008,493**     | **+1,192,726**               | skrub-full     |
| introverts-extroverts-classification | acc ↑     | 0.9663         | **0.9684**        | +0.0022                      | skrub-full     |
| spaceship-titanic                    | acc ↑     | **0.8223**     | 0.8091            | −0.0132                      | vanilla        |
| multi-class-pred-obesity-risk        | acc ↑     | **0.9068**     | 0.9063            | **−0.0005**                  | vanilla (~tie) |
| abalone-regression                   | RMSLE ↓   | **0.1486**     | 0.1510            | **−0.0025**                  | vanilla (~tie) |
| blueberry-yield-regression           | MAE ↓     | **337.95**     | 340.20            | −2.25                        | vanilla        |
| reservation-cancel-classification    | roc_auc ↑ | **0.8988**     | 0.8230            | −0.0758                      | vanilla        |
| covid19-forecasting-regression       | RMSLE ↓   | **0.7677**     | 0.8449            | −0.0772                      | vanilla        |
| employee-attrition-classification    | roc_auc ↑ | **0.8329**     | 0.6180            | −0.2149                      | vanilla        |


Full table: `[report.md` → "Vanilla vs skrub-full"](automated_evaluation/eval_results/20260707_115040_full/report.md).

**Takeaways.**

- Where DataOps structure helps, it helps a lot: **bike-sharing** ([run](automated_evaluation/runs/20260707_115040_full/bike-sharing-regression/skrub-full/gpt-5.4-mini/run1/)) and **restaurant-revenue** ([run](automated_evaluation/runs/20260707_115040_full/restaurant-revenue-regression/skrub-full/gpt-5.4-mini/run1/)) — both heavy on categorical/text and datetime columns, exactly what `TableVectorizer`/DataOps encoders can target well with defaults.
- The one clear regression is **employee-attrition** (0.83 → 0.62 roc_auc): the skrub-full solution underperformed badly on a small, imbalanced table — a concrete case we investigated (see [caveats](#threats-to-validity)).
- Two "losses" (obesity −0.0005, abalone −0.0025) are within noise for N = 1.
- Overall, our skrub-full version performed reasonably well in terms of validation scores, even though the agents are more restricted (i.e. cannot use sklearn-only pipelines, which the LLMs are heavily trained on). They must produce DataOps pipelines and follow increased prompt and code-check guidelines - which can increase debugging effort/drift and distract from pipeline optimization. (follow-up in discussion)

## Which stage produced the winning solution?

For each run we can see which stage's script was promoted as the submission source (`Src val`, by MLE-Star logic the best produced script during runtime) and how each stage moved the holdout score (per-run table in `[report.md](automated_evaluation/eval_results/20260707_115040_full/report.md)`).

### Refinement stage

We added our `skrub.TableReport` novelty in refinement, to feed the ablation agent detailed but compact dataset-summary to make ablation more targeted and therefore imporve refinement. Refinement improved the holdout score over initialization in **5/10** skrub-full tasks:


| Task                                 | Init → Refine              | Improvement          |
| ------------------------------------ | -------------------------- | -------------------- |
| bike-sharing-regression              | 0.0248 → 0.0194 RMSLE      | −0.0054 (~22% lower) |
| restaurant-revenue-regression        | 3,402,997 → 3,096,727 RMSE | −306,270 (~9% lower) |
| blueberry-yield-regression           | 345.31 → 341.53 MAE        | −3.78 (~1%)          |
| employee-attrition-classification    | 0.6175 → 0.6180 roc_auc    | +0.0005              |
| introverts-extroverts-classification | 0.9682 → 0.9684 acc        | +0.0002              |


In the other 5 tasks refinement held the init solution (no positive-gain variant was promoted — exactly what the [robust promotion guard](CONTRIBUTIONS.md#10-agent-level-guards-empty-tool-call--promotion-correctness) is designed to do: keep the previous solution rather than regress).

However, refinement is rarely the *final* winner: **ensemble** produced the promoted submission source in most skrub-full tasks (bike, blueberry, covid, reservation, restaurant, spaceship — strictly better than refine; employee/introverts tie ensemble). Refinement was the top pre-submission stage only where the ensemble regressed (e.g. **obesity**, where init = refine and ensemble was worse). So the biggest score jumps came from **init → refine → ensemble** together, with ensemble usually landing the final gain. This is also the main design goal of MLE-star's agents; each stage should improve the previously promoted solution further, else it would be wasted LLM calls.

*(For contrast, vanilla promoted the refinement structural winner more often — bike, blueberry, employee, obesity — because its ensemble stage frequently regressed, e.g. bike ensemble 1.098 vs refine 0.2991.)*

### Tuning stage (new, improved-only)

The dedicated tuning stage was **triggered in only 1/10 tasks** (blueberry-yield, [run](automated_evaluation/runs/20260707_115040_full/blueberry-yield-regression/skrub-full/gpt-5.4-mini/run1/); see the "Skrub-full extras" table in `[report.md](automated_evaluation/eval_results/20260707_115040_full/report.md)`), and in that run it produced MAE **341.82**, *worse* than both refinement (341.53) and the ensemble winner (340.20).

- **Tuning promotions: 0/1 (0/10 overall).** The [promotion gate](CONTRIBUTIONS.md#12-new-tuning-stage-sub_agentstuning-improved-only) correctly did **not** promote the tuned result because it did not beat the structural winner, i.e. the guard prevented a regression rather than adding a win.
- This is the weakest-evidenced part of the pipeline: the stage seldom fires and hasn't yet demonstrated a net gain on this benchmark/model. It's a prime target for the [future work](#future-enhancements) (looser trigger conditions, larger `tuning_n_iter`, different model).
- Because we already observed this in prior trials, we did instruct the tuning agent to **decide itself** wether tuning should be skipped based on ablation signal (ablation summary wired). If the agent decides ablation and refinement already squeezed out most of the potential gains and tuning is only marginal, then it's skipped.

## Secondary metrics (efficiency & structure)

### DataOps adherence — the structural goal

Our core objective — make generated code DataOps-native — is clearly met:

- **skrub-full DataOps adherence: 0.82–0.94** across tasks; **vanilla: 0.0** everywhere (see `DataOps mean` in `[report.md](automated_evaluation/eval_results/20260707_115040_full/report.md)`).
- Skill tool usage was active: **75–166** `list_skills`/`load_skill`* calls per task, confirming the on-demand skill (not prompt bloat) is what steers structure.

### Compute vs wall-clock time (per task, per system)

Two timing axes matter (see [metric guide](#timing)): **Exec** = summed Python script `execution_time` (pure compute, no LLM), **Wall** = end-to-end run time (LLM + scripts). The gap `Wall − Exec` is essentially LLM "thinking" (planning + skill calls). Source: [`report.md` → "Per (task, system) summary"](automated_evaluation/eval_results/20260707_115040_full/report.md). Δ columns are `vanilla − skrub` → **positive = skrub-full is more efficient**; DataOps mean is `[0,1]` (vanilla = 0.0 on every task).

| Task | Wall skrub (s) | Wall vanilla (s) | ΔWall | Exec skrub (s) | Exec vanilla (s) | ΔExec | DataOps skrub |
|---|---:|---:|---:|---:|---:|---:|---:|
| abalone-regression | 2,362 | 5,506 | **+3,143** | 2,092 | 2,872 | **+780** | 0.861 |
| bike-sharing-regression | 1,251 | 977 | −274 | 588 | 842 | **+254** | 0.944 |
| blueberry-yield-regression | 9,673 | 1,045 | −8,628 | 896 | 1,021 | **+124** | 0.821 |
| covid19-forecasting-regression | 11,348 | 1,001 | −10,347 | 2,004 | 358 | −1,646 | 0.931 |
| employee-attrition-classification | 804 | 226 | −578 | 771 | 89 | −683 | 0.917 |
| introverts-extroverts-classification | 1,014 | 842 | −172 | 297 | 599 | **+302** | 0.917 |
| multi-class-pred-obesity-risk | 3,554 | 6,112 | **+2,558** | 1,322 | 3,388 | **+2,066** | 0.833 |
| reservation-cancel-classification | 1,671 | 3,740 | **+2,069** | 1,695 | 2,184 | **+490** | 0.861 |
| restaurant-revenue-regression | 538 | 280 | −258 | 131 | 216 | **+85** | 0.875 |
| spaceship-titanic | 1,237 | 2,015 | **+778** | 566 | 2,170 | **+1,604** | 0.847 |

### Where skrub-full gained efficiency (and where it didn't)

**Compute (Exec) — usually cheaper.** skrub-full's scripts run faster in **8/10** tasks. The biggest compute savings are on the heaviest vanilla runs: **obesity** (−2,066s, ~61% less script time) and **spaceship** (−1,604s, ~74% less). DataOps pipelines plus our runtime rules (silenced booster logs, preview subsampling, no redundant full-train/export in early stages — see [deferred submission export](CONTRIBUTIONS.md#9-prompt-hardening-for-skrub-dataops-all-stages)) keep per-script compute down. The two exceptions are **covid19** (−1,646s) and **employee** (−683s), where skrub-full built heavier pipelines.

**Wall-clock — big wins on the expensive tasks, big losses when extra stages fire.** skrub-full is faster end-to-end in **4/10** tasks, but those wins land exactly where vanilla was slowest: **abalone** (−57% wall), **reservation** (−55%), **obesity** (−42%), **spaceship** (−39%). The losses are dominated by LLM overhead, not compute:

| Task | Wall | Exec | LLM overhead (Wall−Exec) | Why |
|---|---:|---:|---:|---|
| covid19 (skrub) | 11,348 | 2,004 | **9,344** | 166 skill calls (most in batch) |
| blueberry (skrub) | 9,673 | 896 | **8,777** | tuning stage ran + skill calls |

For comparison, vanilla's covid overhead is only ~643s (1,001 − 358). So when the skill/tuning/TableReport stages engage heavily, wall time is spent "thinking", not computing.

**Net.** The improved pipeline is **computationally leaner per script (8/10)** and **faster end-to-end on the most expensive tasks (4/10, all large vanilla runs)**, at the cost of extra LLM round-trips when skill loading and the tuning stage fire. Per-stage breakdown (which stage the time goes to) is in [`report.md` → "Stage script execution"](automated_evaluation/eval_results/20260707_115040_full/report.md); the clearest stage-level shifts are spaceship (ensemble exec 120s skrub vs 684s vanilla; submit 43s vs 384s) and obesity (init exec 395s skrub vs 1,423s vanilla).

## Connecting results to our modifications


| Modification                                                              | Evidence in this batch                                                                                                       |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| skrub DataOps skill (§2, §8, §13 in [CONTRIBUTIONS.md](CONTRIBUTIONS.md)) | DataOps adherence 0.82–0.94 vs 0; large wins on text/categorical/datetime tasks (bike, restaurant)                           |
| Robust promotion guards (§10)                                             | Refinement never regressed init (kept previous solution in 5/10); ensemble regressions in vanilla show why the guard matters |
| Tuning stage + promotion gate (§6, §12)                                   | Fired 1/10; gate correctly withheld a worse tuned result (0 bad promotions)                                                  |
| Execution-robustness gates (§5)                                           | 20/20 runs produced usable `final_state.json`; no empty/tool-call-only turn scored as valid                                  |
| TableReport profiling (§3, §11)                                           | Wired into refinement planners; qualitative (contribution to structural wins), not isolated here                             |
| Runtime compat (§1)                                                       | Enabled the whole benchmark to run on `openai/gpt-5.4-mini` at all                                                           |


Net: the improvements **reliably deliver the structural/DataOps objective** and can win big on the right task types, but on this single-model, single-repeat run they did **not** yield a net primary-metric win over vanilla, and the tuning stage is under-exercised.

## Discussion
TODO
- model drift submission (vs. vanilla, full submission also early)
- model drift generally and guards
- skrub skill vs without
- efficiency gains and where not

## Summary
TODO

## Threats to validity

- **N = 1, single seed (42), single model.** No variance estimates; individual results (esp. employee-attrition) may be run-specific.
- **Holdout ≠ leaderboard.** Scores are internal validation prints, not Kaggle test scores; submission-print anomalies (bike-sharing sub = 0, covid sub better than upstream) mean **(best)** is the more trustworthy column for those tasks.
- **Ensemble dominance** means refinement/tuning gains can be masked by the final ensemble step.
- **Tuning under-triggered**, so its effect is essentially untested here.

## Future enhancements

- **Kaggle / test-set scoring.** Submit each task's `./final/submission.csv` to obtain real leaderboard scores and compare against holdout (validate that DataOps wins transfer to the test set). *Placeholder — not yet run.*
- **Additional models.** Re-run the full matrix with another LLM (e.g. `mistral-large-`*) into the same or a sibling runs root (see [EXPERIMENTS.md → Adding another model](EXPERIMENTS.md#adding-another-model-eg-mistral)); the report groups per model automatically. *Placeholder.*
- **Repeats for significance.** Add `run2`/`run3` with different seeds to get mean ± std (see [EXPERIMENTS.md → Adding another repeat](EXPERIMENTS.md#adding-another-repeat-eg-run2)).
- **Tuning stage tuning.** Investigate why it triggers rarely; try looser trigger conditions / larger `tuning_n_iter` and measure promotion rate.
- **Investigate employee-attrition regression** (0.83 → 0.62 roc_auc) — likely a DataOps encoding/holdout issue on a small imbalanced table.

---

## Metric guide

Reports are generated with:

```bash
python3 automated_evaluation/evaluate.py \
  --summarize-only \
  --runs-root automated_evaluation/runs/<batch-stamp>
```

Output: `automated_evaluation/eval_results/<batch-stamp>/report.md`.

Each task uses the metric named in its task pack (e.g. `accuracy_score`, `roc_auc_score`, `root_mean_squared_log_error`); lower-is-better vs higher-is-better follows the task definition.

### Validation scores (holdout)

All val scores come from the `Final Validation Performance:` line in executed pipeline scripts, parsed from `final_state.json` exec results — not from test-set leaderboard scores.


| Name                 | Report label             | Meaning                                                                                                                                                                       |
| -------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Submission score** | **(sub)**                | Holdout metric from the submission agent run (`final_solution.py` / `submission_code_exec_result`).                                                                           |
| **Upstream score**   | **(best)**               | Holdout metric of the script passed *into* the submission agent: the best structural/ensemble solution before export. Prefer this when the submission print looks suspicious. |
| **Primary score**    | `Best` in per-run tables | Best holdout score across **all** stages (init → refine → tune → ensemble → submission).                                                                                      |
| **Source val**       | `Src val`                | Per-run alias for the upstream **(best)** score.                                                                                                                              |


### Per-stage val scores (per-run table)


| Column   | Stage                                                          |
| -------- | -------------------------------------------------------------- |
| Init     | Best holdout score after initialization                        |
| Refine   | Score promoted after refinement (structural outer-loop winner) |
| Tune     | Score after tuning bake (`-` if tuning did not run)            |
| Ensemble | Best holdout score from the ensemble stage                     |
| Gain     | Signed improvement from init to primary score                  |


### Timing


| Column                                    | Meaning                                                                                            |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Init/Refine/Tune/Ensemble/Submit exec** | Sum of Python `execution_time` for scripts in that stage. Script time only — excludes LLM latency. |
| **Wall total**                            | End-to-end run wall-clock (`meta.json` `started_at` → `finished_at`). LLM + scripts.               |


### Skrub-full only


| Column          | Meaning                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| **Tuning ran**  | Whether the tuning stage executed (not skipped).                                                         |
| **Skill calls** | Total ADK log invocations of `list_skills` / `load_skill` / `load_skill_resource`.                       |
| **DataOps**     | Share `[0, 1]` of core skrub DataOps patterns across produced `.py` scripts. Vanilla typically scores 0. |


