# MLE-STAR automated evaluation report

- Generated (UTC): 2026-07-09T11:11:13.046724+00:00
- Git SHA: `1493edc`
- Runs root: `automated_evaluation/runs/20260707_115040_full`
- Seed: `42`
- Model: `openai/gpt-5.4-mini`

## Scanned tasks

Complete: **10 / 10**

| Task | Type | Metric | Ready |
| --- | --- | --- | --- |
| abalone-regression | Tabular Regression | root_mean_squared_log_error | yes |
| bike-sharing-regression | Tabular Regression | root_mean_squared_log_error | yes |
| blueberry-yield-regression | Tabular Regression | mean_absolute_error | yes |
| covid19-forecasting-regression | Tabular Regression | root_mean_squared_log_error | yes |
| employee-attrition-classification | Tabular Classification | roc_auc_score | yes |
| introverts-extroverts-classification | Tabular Classification | accuracy_score | yes |
| multi-class-pred-obesity-risk | Tabular Classification | accuracy_score | yes |
| reservation-cancel-classification | Tabular Classification | roc_auc_score | yes |
| restaurant-revenue-regression | Tabular Regression | root_mean_squared_error | yes |
| spaceship-titanic | Tabular Classification | accuracy_score | yes |

**Column guide:**
- **Task** — Task folder name under `tasks/`.
- **Type** — Problem type (regression or classification).
- **Metric** — Primary metric named in the task pack (e.g. RMSE, accuracy).
- **Ready** — `yes` if `train.csv`, `test.csv`, and `task_description.txt` exist.

## Archived runs analyzed

Total runs with usable `final_state.json`: **20**

## Vanilla vs skrub-full (openai/gpt-5.4-mini)

_**(sub)** = holdout score from the submission agent's `final_solution.py` run. **(best)** = holdout score of the upstream script the submission agent received (best structural/ensemble solution before export). `primary_score` in per-run tables is the best score across all stages, not shown here._

| Task | Metric | Vanilla (sub) | Skrub-full (sub) | Delta (sub) | Vanilla (best) | Skrub-full (best) | Delta (best) | Winner (best) | Winner (sub) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | root_mean_squared_log_error | 0.1486 | 0.151 | -0.0025 | 0.1486 | 0.151 | -0.0025 | vanilla | vanilla |
| bike-sharing-regression | root_mean_squared_log_error | - | 0 | - | 0.2991 | 0.0144 | 0.2846 | skrub-full | - |
| blueberry-yield-regression | mean_absolute_error | 337.9509 | 340.199 | -2.2481 | 337.9509 | 340.199 | -2.2481 | vanilla | vanilla |
| covid19-forecasting-regression | root_mean_squared_log_error | 0.7677 | 0.4543 | 0.3135 | 0.7677 | 0.8449 | -0.0772 | vanilla | skrub-full |
| employee-attrition-classification | roc_auc_score | 0.8307 | 0.618 | -0.2127 | 0.8329 | 0.618 | -0.2149 | vanilla | vanilla |
| introverts-extroverts-classification | accuracy_score | 0.966 | 0.9684 | 0.0024 | 0.9663 | 0.9684 | 0.0022 | skrub-full | skrub-full |
| multi-class-pred-obesity-risk | accuracy_score | - | 0.9063 | - | 0.9068 | 0.9063 | -0.0005 | vanilla | - |
| reservation-cancel-classification | roc_auc_score | 0.8988 | 0.823 | -0.0758 | 0.8988 | 0.823 | -0.0758 | vanilla | vanilla |
| restaurant-revenue-regression | root_mean_squared_error | - | 2008493.1393 | - | 3201219.1124 | 2008493.1393 | 1192725.9731 | skrub-full | - |
| spaceship-titanic | accuracy_score | 0.8223 | 0.7849 | -0.0374 | 0.8223 | 0.8091 | -0.0132 | vanilla | vanilla |

**Column guide:**
- **Task** — Benchmark task name.
- **Metric** — Validation metric for this task.
- **Vanilla (sub)** — Mean holdout score printed by the submission agent script (`final_solution.py` / `submission_code_exec_result`).
- **Skrub-full (sub)** — Mean submission-stage holdout score for skrub-full repeats.
- **Delta (sub)** — Signed skrub advantage on submission scores (positive ⇒ skrub better).
- **Vanilla (best)** — Mean holdout score of the upstream script passed into the submission agent (best structural/ensemble solution before export).
- **Skrub-full (best)** — Mean upstream-script holdout score for skrub-full repeats.
- **Delta (best)** — Signed skrub advantage on upstream-script scores (positive ⇒ skrub better).
- **Winner (best)** — System with better mean upstream-script score (`vanilla`, `skrub-full`, or `tie`).
- **Winner (sub)** — System with better mean submission-script score.
Upstream (best) — Skrub-full wins: **3/10** · Vanilla wins: **7/10** · Ties: **0/10**
Submission (sub) — Skrub-full wins: **2/10** · Vanilla wins: **5/10** · Ties: **0/10**

## Per (task, system) summary (openai/gpt-5.4-mini)

| Task | System | Model | Metric | N | Score mean | Score std | Wall mean (s) | Exec mean (s) | DataOps mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_log_error | 1 | 0.151 | - | 2362.3 | 2092.1 | 0.861 |
| abalone-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_log_error | 1 | 0.1486 | - | 5505.5 | 2872.3 | 0 |
| bike-sharing-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_log_error | 1 | 0 | - | 1251 | 588.4 | 0.944 |
| bike-sharing-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_log_error | 1 | 0.2991 | - | 976.9 | 842.1 | 0 |
| blueberry-yield-regression | skrub-full | openai/gpt-5.4-mini | mean_absolute_error | 1 | 340.199 | - | 9672.9 | 896.4 | 0.821 |
| blueberry-yield-regression | vanilla | openai/gpt-5.4-mini | mean_absolute_error | 1 | 337.9509 | - | 1044.6 | 1020.5 | 0 |
| covid19-forecasting-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_log_error | 1 | 0.4543 | - | 11347.8 | 2004.2 | 0.931 |
| covid19-forecasting-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_log_error | 1 | 0.7677 | - | 1000.5 | 358.2 | 0 |
| employee-attrition-classification | skrub-full | openai/gpt-5.4-mini | roc_auc_score | 1 | 0.618 | - | 804.3 | 771.2 | 0.917 |
| employee-attrition-classification | vanilla | openai/gpt-5.4-mini | roc_auc_score | 1 | 0.8329 | - | 226.3 | 88.5 | 0 |
| introverts-extroverts-classification | skrub-full | openai/gpt-5.4-mini | accuracy_score | 1 | 0.9684 | - | 1013.9 | 296.5 | 0.917 |
| introverts-extroverts-classification | vanilla | openai/gpt-5.4-mini | accuracy_score | 1 | 0.9663 | - | 842.1 | 598.5 | 0 |
| multi-class-pred-obesity-risk | skrub-full | openai/gpt-5.4-mini | accuracy_score | 1 | 0.9063 | - | 3554.1 | 1321.7 | 0.833 |
| multi-class-pred-obesity-risk | vanilla | openai/gpt-5.4-mini | accuracy_score | 1 | 0.9068 | - | 6111.9 | 3387.5 | 0 |
| reservation-cancel-classification | skrub-full | openai/gpt-5.4-mini | roc_auc_score | 1 | 0.823 | - | 1670.8 | 1694.5 | 0.861 |
| reservation-cancel-classification | vanilla | openai/gpt-5.4-mini | roc_auc_score | 1 | 0.8988 | - | 3739.5 | 2184.3 | 0 |
| restaurant-revenue-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_error | 1 | 2008493.1393 | - | 538.4 | 130.7 | 0.875 |
| restaurant-revenue-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_error | 1 | 3201219.1124 | - | 280.3 | 216.1 | 0 |
| spaceship-titanic | skrub-full | openai/gpt-5.4-mini | accuracy_score | 1 | 0.8091 | - | 1236.8 | 565.7 | 0.847 |
| spaceship-titanic | vanilla | openai/gpt-5.4-mini | accuracy_score | 1 | 0.8223 | - | 2015.2 | 2169.7 | 0 |

**Column guide:**
- **Task** — Benchmark task name.
- **System** — Agent variant aggregated (`vanilla` or `skrub-full`).
- **Model** — LLM model slug from the run archive path / state.
- **Metric** — Validation metric for this task.
- **N** — Number of archived runs in this group.
- **Score mean** — Mean `score_final_best` across repeats.
- **Score std** — Standard deviation of primary scores across repeats (`-` when *N* = 1).
- **Wall mean (s)** — Mean end-to-end wall-clock time per run.
- **Exec mean (s)** — Mean total Python script execution time per run.
- **DataOps mean** — Mean core DataOps adherence in `[0, 1]` across produced scripts.

### Stage script execution — mean per (task, system) (openai/gpt-5.4-mini)

_Per-stage times sum Python script `execution_time` from `final_state.json` (excludes LLM latency). Wall total includes LLM + scripts._

| Task | System | Metric | Init exec | Refine exec | Tune exec | Ensemble exec | Submit exec | Exec total | Wall total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | root_mean_squared_log_error | 639.1 | 666.2 | 0 | 379 | 407.8 | 2092.1 | 2362.3 |
| abalone-regression | vanilla | root_mean_squared_log_error | 341.8 | 1124.1 | 0 | 978.1 | 428.3 | 2872.3 | 5505.5 |
| bike-sharing-regression | skrub-full | root_mean_squared_log_error | 371.9 | 150.4 | 0 | 57.7 | 8.5 | 588.4 | 1251 |
| bike-sharing-regression | vanilla | root_mean_squared_log_error | 109.3 | 401.3 | 0 | 235.4 | 96.1 | 842.1 | 976.9 |
| blueberry-yield-regression | skrub-full | mean_absolute_error | 377.5 | 163.9 | 194.8 | 83 | 77.3 | 896.4 | 9672.9 |
| blueberry-yield-regression | vanilla | mean_absolute_error | 129.9 | 358.9 | 0 | 478.5 | 53.1 | 1020.5 | 1044.6 |
| covid19-forecasting-regression | skrub-full | root_mean_squared_log_error | 558.4 | 376.3 | 0 | 473.3 | 596.2 | 2004.2 | 11347.8 |
| covid19-forecasting-regression | vanilla | root_mean_squared_log_error | 116.6 | 155.3 | 0 | 59.2 | 27.1 | 358.2 | 1000.5 |
| employee-attrition-classification | skrub-full | roc_auc_score | 89.9 | 490.4 | 0 | 146.8 | 44.1 | 771.2 | 804.3 |
| employee-attrition-classification | vanilla | roc_auc_score | 27.8 | 48 | 0 | 9.5 | 3.2 | 88.5 | 226.3 |
| introverts-extroverts-classification | skrub-full | accuracy_score | 88.9 | 84.5 | 0 | 81 | 42.2 | 296.5 | 1013.9 |
| introverts-extroverts-classification | vanilla | accuracy_score | 134.1 | 100.6 | 0 | 246 | 117.9 | 598.5 | 842.1 |
| multi-class-pred-obesity-risk | skrub-full | accuracy_score | 394.7 | 443.1 | 0 | 160.6 | 323.2 | 1321.7 | 3554.1 |
| multi-class-pred-obesity-risk | vanilla | accuracy_score | 1422.6 | 1001.3 | 0 | 720.2 | 243.5 | 3387.5 | 6111.9 |
| reservation-cancel-classification | skrub-full | roc_auc_score | 422.8 | 701.5 | 0 | 303.9 | 266.3 | 1694.5 | 1670.8 |
| reservation-cancel-classification | vanilla | roc_auc_score | 484.2 | 1474.3 | 0 | 170.9 | 54.9 | 2184.3 | 3739.5 |
| restaurant-revenue-regression | skrub-full | root_mean_squared_error | 34 | 42.6 | 0 | 27.7 | 26.4 | 130.7 | 538.4 |
| restaurant-revenue-regression | vanilla | root_mean_squared_error | 85.8 | 58.7 | 0 | 43.7 | 28 | 216.1 | 280.3 |
| spaceship-titanic | skrub-full | accuracy_score | 179.1 | 223.2 | 0 | 120 | 43.4 | 565.7 | 1236.8 |
| spaceship-titanic | vanilla | accuracy_score | 473.8 | 627.3 | 0 | 684.3 | 384.4 | 2169.7 | 2015.2 |

**Column guide:**
- **Task** — Benchmark task name.
- **System** — Agent variant.
- **Metric** — Validation metric.
- **Init exec** — Mean script execution time in initialization (s).
- **Refine exec** — Mean script execution time in refinement (s).
- **Tune exec** — Mean script execution time in tuning (s).
- **Ensemble exec** — Mean script execution time in ensembling (s).
- **Submit exec** — Mean script execution time in submission (s).
- **Exec total** — Mean total script execution time (s).
- **Wall total** — Mean end-to-end wall-clock time (s).

### Skrub-full extras (openai/gpt-5.4-mini)

| Task | Metric | Tuning ran | Skill calls |
| --- | --- | --- | --- |
| abalone-regression | root_mean_squared_log_error | no | 82 |
| bike-sharing-regression | root_mean_squared_log_error | no | 89 |
| blueberry-yield-regression | mean_absolute_error | yes | 88 |
| covid19-forecasting-regression | root_mean_squared_log_error | no | 166 |
| employee-attrition-classification | roc_auc_score | no | 81 |
| introverts-extroverts-classification | accuracy_score | no | 143 |
| multi-class-pred-obesity-risk | accuracy_score | no | 100 |
| reservation-cancel-classification | roc_auc_score | no | 75 |
| restaurant-revenue-regression | root_mean_squared_error | no | 99 |
| spaceship-titanic | accuracy_score | no | 104 |

**Column guide:**
- **Task** — Benchmark task name.
- **Metric** — Validation metric for this task.
- **Tuning ran** — `yes` if the tuning stage executed for any repeat of this task.
- **Skill calls** — Total skill tool invocations (`list_skills` / `load_skill*`) in ADK logs.

## Per-run results (openai/gpt-5.4-mini)

| Task | System | Model | Metric | Run | Best | Src val | Init | Refine | Tune | Ensemble | Gain | DataOps | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_log_error | run1 | 0.151 | 0.151 | 0.151 | 0.151 | - | 0.151 | 0 | 0.861 | 2362.3 |
| abalone-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_log_error | run1 | 0.1486 | 0.1486 | 0.1503 | 0.1486 | - | 0.1486 | 0.0017 | 0 | 5505.5 |
| bike-sharing-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_log_error | run1 | 0 | 0.0144 | 0.0248 | 0.0194 | - | 0.0144 | 0.0248 | 0.944 | 1251 |
| bike-sharing-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_log_error | run1 | 0.2991 | 0.2991 | 0.3552 | 0.2991 | - | 1.098 | 0.0561 | 0 | 976.9 |
| blueberry-yield-regression | skrub-full | openai/gpt-5.4-mini | mean_absolute_error | run1 | 340.199 | 340.199 | 345.3099 | 341.5349 | 341.8231 | 340.199 | 5.1109 | 0.821 | 9672.9 |
| blueberry-yield-regression | vanilla | openai/gpt-5.4-mini | mean_absolute_error | run1 | 337.9509 | 337.9509 | 338.5355 | 337.9509 | - | 357.747 | 0.5846 | 0 | 1044.6 |
| covid19-forecasting-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_log_error | run1 | 0.4543 | 0.8449 | 0.8733 | 0.8733 | - | 0.8449 | 0.419 | 0.931 | 11347.8 |
| covid19-forecasting-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_log_error | run1 | 0.7677 | 0.7677 | 0.7677 | 0.7677 | - | 0.7677 | 0 | 0 | 1000.5 |
| employee-attrition-classification | skrub-full | openai/gpt-5.4-mini | roc_auc_score | run1 | 0.618 | 0.618 | 0.6175 | 0.618 | - | 0.618 | 0.0005 | 0.917 | 804.3 |
| employee-attrition-classification | vanilla | openai/gpt-5.4-mini | roc_auc_score | run1 | 0.8329 | 0.8329 | 0.8293 | 0.8329 | - | 0.8307 | 0.0036 | 0 | 226.3 |
| introverts-extroverts-classification | skrub-full | openai/gpt-5.4-mini | accuracy_score | run1 | 0.9684 | 0.9684 | 0.9682 | 0.9684 | - | 0.9684 | 0.0003 | 0.917 | 1013.9 |
| introverts-extroverts-classification | vanilla | openai/gpt-5.4-mini | accuracy_score | run1 | 0.9663 | 0.9663 | 0.966 | 0.966 | - | 0.9663 | 0.0003 | 0 | 842.1 |
| multi-class-pred-obesity-risk | skrub-full | openai/gpt-5.4-mini | accuracy_score | run1 | 0.9063 | 0.9063 | 0.9063 | 0.9063 | - | 0.8947 | 0 | 0.833 | 3554.1 |
| multi-class-pred-obesity-risk | vanilla | openai/gpt-5.4-mini | accuracy_score | run1 | 0.9068 | 0.9068 | 0.9051 | 0.9068 | - | 0.9066 | 0.0017 | 0 | 6111.9 |
| reservation-cancel-classification | skrub-full | openai/gpt-5.4-mini | roc_auc_score | run1 | 0.823 | 0.823 | 0.8229 | 0.8229 | - | 0.823 | 0.0001 | 0.861 | 1670.8 |
| reservation-cancel-classification | vanilla | openai/gpt-5.4-mini | roc_auc_score | run1 | 0.8988 | 0.8988 | 0.8988 | 0.8988 | - | 0.8988 | 0 | 0 | 3739.5 |
| restaurant-revenue-regression | skrub-full | openai/gpt-5.4-mini | root_mean_squared_error | run1 | 2008493.1393 | 2008493.1393 | 3402996.9195 | 3096727.1315 | - | 2008493.1393 | 1394503.7802 | 0.875 | 538.4 |
| restaurant-revenue-regression | vanilla | openai/gpt-5.4-mini | root_mean_squared_error | run1 | 3201219.1124 | 3201219.1124 | 3342815.2517 | 3323284.6489 | - | 3201219.1124 | 141596.1393 | 0 | 280.3 |
| spaceship-titanic | skrub-full | openai/gpt-5.4-mini | accuracy_score | run1 | 0.8091 | 0.8091 | 0.8074 | 0.8074 | - | 0.8091 | 0.0017 | 0.847 | 1236.8 |
| spaceship-titanic | vanilla | openai/gpt-5.4-mini | accuracy_score | run1 | 0.8223 | 0.8223 | 0.8148 | 0.8217 | - | 0.8223 | 0.0075 | 0 | 2015.2 |

**Column guide:**
- **Task** — Benchmark task name.
- **System** — Agent variant for this run (`vanilla` or `skrub-full`).
- **Model** — LLM model slug.
- **Metric** — Validation metric for this task.
- **Run** — Repeat label (e.g. `run1`).
- **Best** — `score_final_best` across init/refine/tune/ensemble/submission.
- **Src val** — Holdout score of the upstream script chosen for submission export.
- **Init** — Best holdout score after initialization.
- **Refine** — Holdout score promoted after refinement.
- **Tune** — Holdout score after tuning (`-` if tuning did not run).
- **Ensemble** — Best holdout score from the ensemble stage.
- **Gain** — Signed improvement from init to `primary_score`.
- **DataOps** — Core DataOps adherence `[0, 1]`.
- **Wall (s)** — End-to-end wall-clock seconds for this run.

### Stage script execution — per run (openai/gpt-5.4-mini)

| Task | System | Run | Init exec | Refine exec | Tune exec | Ensemble exec | Submit exec | Exec total | Wall total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | run1 | 639.1 | 666.2 | 0 | 379 | 407.8 | 2092.1 | 2362.3 |
| abalone-regression | vanilla | run1 | 341.8 | 1124.1 | 0 | 978.1 | 428.3 | 2872.3 | 5505.5 |
| bike-sharing-regression | skrub-full | run1 | 371.9 | 150.4 | 0 | 57.7 | 8.5 | 588.4 | 1251 |
| bike-sharing-regression | vanilla | run1 | 109.3 | 401.3 | 0 | 235.4 | 96.1 | 842.1 | 976.9 |
| blueberry-yield-regression | skrub-full | run1 | 377.5 | 163.9 | 194.8 | 83 | 77.3 | 896.4 | 9672.9 |
| blueberry-yield-regression | vanilla | run1 | 129.9 | 358.9 | 0 | 478.5 | 53.1 | 1020.5 | 1044.6 |
| covid19-forecasting-regression | skrub-full | run1 | 558.4 | 376.3 | 0 | 473.3 | 596.2 | 2004.2 | 11347.8 |
| covid19-forecasting-regression | vanilla | run1 | 116.6 | 155.3 | 0 | 59.2 | 27.1 | 358.2 | 1000.5 |
| employee-attrition-classification | skrub-full | run1 | 89.9 | 490.4 | 0 | 146.8 | 44.1 | 771.2 | 804.3 |
| employee-attrition-classification | vanilla | run1 | 27.8 | 48 | 0 | 9.5 | 3.2 | 88.5 | 226.3 |
| introverts-extroverts-classification | skrub-full | run1 | 88.9 | 84.5 | 0 | 81 | 42.2 | 296.5 | 1013.9 |
| introverts-extroverts-classification | vanilla | run1 | 134.1 | 100.6 | 0 | 246 | 117.9 | 598.5 | 842.1 |
| multi-class-pred-obesity-risk | skrub-full | run1 | 394.7 | 443.1 | 0 | 160.6 | 323.2 | 1321.7 | 3554.1 |
| multi-class-pred-obesity-risk | vanilla | run1 | 1422.6 | 1001.3 | 0 | 720.2 | 243.5 | 3387.5 | 6111.9 |
| reservation-cancel-classification | skrub-full | run1 | 422.8 | 701.5 | 0 | 303.9 | 266.3 | 1694.5 | 1670.8 |
| reservation-cancel-classification | vanilla | run1 | 484.2 | 1474.3 | 0 | 170.9 | 54.9 | 2184.3 | 3739.5 |
| restaurant-revenue-regression | skrub-full | run1 | 34 | 42.6 | 0 | 27.7 | 26.4 | 130.7 | 538.4 |
| restaurant-revenue-regression | vanilla | run1 | 85.8 | 58.7 | 0 | 43.7 | 28 | 216.1 | 280.3 |
| spaceship-titanic | skrub-full | run1 | 179.1 | 223.2 | 0 | 120 | 43.4 | 565.7 | 1236.8 |
| spaceship-titanic | vanilla | run1 | 473.8 | 627.3 | 0 | 684.3 | 384.4 | 2169.7 | 2015.2 |

**Column guide:**
- **Task** — Benchmark task name.
- **System** — Agent variant.
- **Run** — Repeat label.
- **Init exec** — Summed Python script execution time during initialization.
- **Refine exec** — Summed Python script execution time during refinement.
- **Tune exec** — Summed Python script execution time during tuning.
- **Ensemble exec** — Summed Python script execution time during ensembling.
- **Submit exec** — Summed Python script execution time during submission export.
- **Exec total** — Total Python script execution time across all stages.
- **Wall total** — End-to-end wall-clock seconds for the full run (LLM + scripts).
