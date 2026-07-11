# MLE-STAR automated evaluation report

- Generated (UTC): 2026-07-10T09:12:26.076032+00:00
- Git SHA: `36c8777`
- Runs root: `automated_evaluation/runs/20260709_144250_gpt_large`
- Seed: `42`
- Model: `openai/gpt-5.4`

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

## Vanilla vs skrub-full (openai/gpt-5.4)

_**(sub)** = holdout score from the submission agent's `final_solution.py` run. **(best)** = holdout score of the upstream script the submission agent received (best structural/ensemble solution before export). `primary_score` in per-run tables is the best score across all stages, not shown here._

| Task | Metric | Vanilla (sub) | Skrub-full (sub) | Delta (sub) | Vanilla (best) | Skrub-full (best) | Delta (best) | Winner (best) | Winner (sub) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | root_mean_squared_log_error | 0.1506 | 0.1507 | -0.0001 | 0.1506 | 0.1507 | -0.0001 | vanilla | vanilla |
| bike-sharing-regression | root_mean_squared_log_error | 0.284 | 0.3171 | -0.0331 | 0.284 | 0.0101 | 0.274 | skrub-full | vanilla |
| blueberry-yield-regression | mean_absolute_error | 336.6524 | 337.4847 | -0.8323 | 336.6524 | 337.4847 | -0.8323 | vanilla | vanilla |
| covid19-forecasting-regression | root_mean_squared_log_error | 0.0547 | 1.1064 | -1.0517 | 0.0547 | 0.2965 | -0.2418 | vanilla | vanilla |
| employee-attrition-classification | roc_auc_score | 0.8139 | 0.8308 | 0.0169 | 0.816 | 0.8331 | 0.0171 | skrub-full | skrub-full |
| introverts-extroverts-classification | accuracy_score | 0.9694 | 0.9722 | 0.0028 | 0.9695 | 0.9725 | 0.003 | skrub-full | skrub-full |
| multi-class-pred-obesity-risk | accuracy_score | 0.9032 | 0.9104 | 0.0072 | 0.9094 | 0.9104 | 0.001 | skrub-full | skrub-full |
| reservation-cancel-classification | roc_auc_score | - | 0.8981 | - | 0.8997 | 0.8981 | -0.0016 | vanilla | - |
| restaurant-revenue-regression | root_mean_squared_error | 2373422.6428 | 3138027.7682 | -764605.1254 | 2373422.6428 | 3131428.1353 | -758005.4925 | vanilla | vanilla |
| spaceship-titanic | accuracy_score | - | 0.8148 | - | 0.82 | 0.8171 | -0.0029 | vanilla | - |

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
Upstream (best) — Skrub-full wins: **4/10** · Vanilla wins: **6/10** · Ties: **0/10**
Submission (sub) — Skrub-full wins: **3/10** · Vanilla wins: **5/10** · Ties: **0/10**

## Per (task, system) summary (openai/gpt-5.4)

| Task | System | Model | Metric | N | Score mean | Score std | Wall mean (s) | Exec mean (s) | DataOps mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_log_error | 1 | 0.1507 | - | 573.4 | 347.1 | 0.847 |
| abalone-regression | vanilla | openai/gpt-5.4 | root_mean_squared_log_error | 1 | 0.1506 | - | 6350.8 | 3274.8 | 0 |
| bike-sharing-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_log_error | 1 | 0.0101 | - | 536.8 | 162.7 | 0.978 |
| bike-sharing-regression | vanilla | openai/gpt-5.4 | root_mean_squared_log_error | 1 | 0.284 | - | 312.7 | 205.1 | 0 |
| blueberry-yield-regression | skrub-full | openai/gpt-5.4 | mean_absolute_error | 1 | 337.4847 | - | 566.9 | 352.3 | 0.861 |
| blueberry-yield-regression | vanilla | openai/gpt-5.4 | mean_absolute_error | 1 | 336.6524 | - | 818.1 | 579.7 | 0 |
| covid19-forecasting-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_log_error | 1 | 0.2965 | - | 1271.7 | 740.3 | 0.857 |
| covid19-forecasting-regression | vanilla | openai/gpt-5.4 | root_mean_squared_log_error | 1 | 0.0547 | - | 893.2 | 660 | 0 |
| employee-attrition-classification | skrub-full | openai/gpt-5.4 | roc_auc_score | 1 | 0.8331 | - | 502.5 | 211.7 | 0.857 |
| employee-attrition-classification | vanilla | openai/gpt-5.4 | roc_auc_score | 1 | 0.816 | - | 471.6 | 291.1 | 0 |
| introverts-extroverts-classification | skrub-full | openai/gpt-5.4 | accuracy_score | 1 | 0.9725 | - | 1101.6 | 377.4 | 0.819 |
| introverts-extroverts-classification | vanilla | openai/gpt-5.4 | accuracy_score | 1 | 0.9695 | - | 867 | 831.1 | 0 |
| multi-class-pred-obesity-risk | skrub-full | openai/gpt-5.4 | accuracy_score | 1 | 0.9104 | - | 2501 | 677.9 | 0.75 |
| multi-class-pred-obesity-risk | vanilla | openai/gpt-5.4 | accuracy_score | 1 | 0.9094 | - | 10810.4 | 4221.3 | 0 |
| reservation-cancel-classification | skrub-full | openai/gpt-5.4 | roc_auc_score | 1 | 0.8981 | - | 1030 | 645.9 | 0.694 |
| reservation-cancel-classification | vanilla | openai/gpt-5.4 | roc_auc_score | 1 | 0.8997 | - | 2146.9 | 1581.1 | 0 |
| restaurant-revenue-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_error | 1 | 3131428.1353 | - | 920.2 | 402.6 | 0.798 |
| restaurant-revenue-regression | vanilla | openai/gpt-5.4 | root_mean_squared_error | 1 | 2373422.6428 | - | 867.1 | 799.7 | 0 |
| spaceship-titanic | skrub-full | openai/gpt-5.4 | accuracy_score | 1 | 0.8171 | - | 529.2 | 245.1 | 0.929 |
| spaceship-titanic | vanilla | openai/gpt-5.4 | accuracy_score | 1 | 0.82 | - | 277.9 | 124.9 | 0 |

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

### Stage script execution — mean per (task, system) (openai/gpt-5.4)

_Per-stage times sum Python script `execution_time` from `final_state.json` (excludes LLM latency). Wall total includes LLM + scripts._

| Task | System | Metric | Init exec | Refine exec | Tune exec | Ensemble exec | Submit exec | Exec total | Wall total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | root_mean_squared_log_error | 87 | 63.3 | 0 | 98.6 | 98.3 | 347.1 | 573.4 |
| abalone-regression | vanilla | root_mean_squared_log_error | 632.5 | 1534.4 | 0 | 745.6 | 362.3 | 3274.8 | 6350.8 |
| bike-sharing-regression | skrub-full | root_mean_squared_log_error | 27.1 | 17.2 | 33.5 | 43.7 | 41.2 | 162.7 | 536.8 |
| bike-sharing-regression | vanilla | root_mean_squared_log_error | 154 | 19.7 | 0 | 19.4 | 12 | 205.1 | 312.7 |
| blueberry-yield-regression | skrub-full | mean_absolute_error | 86.2 | 136.2 | 0 | 63.4 | 66.5 | 352.3 | 566.9 |
| blueberry-yield-regression | vanilla | mean_absolute_error | 97.7 | 210 | 0 | 178.6 | 93.3 | 579.7 | 818.1 |
| covid19-forecasting-regression | skrub-full | root_mean_squared_log_error | 64.6 | 62.6 | 538.8 | 71.1 | 3.2 | 740.3 | 1271.7 |
| covid19-forecasting-regression | vanilla | root_mean_squared_log_error | 130.1 | 303 | 0 | 199.4 | 27.4 | 660 | 893.2 |
| employee-attrition-classification | skrub-full | roc_auc_score | 50.1 | 66.3 | 45.4 | 24.9 | 25.1 | 211.7 | 502.5 |
| employee-attrition-classification | vanilla | roc_auc_score | 55.8 | 49 | 0 | 125 | 61.3 | 291.1 | 471.6 |
| introverts-extroverts-classification | skrub-full | accuracy_score | 72.7 | 74.3 | 0 | 124.5 | 105.9 | 377.4 | 1101.6 |
| introverts-extroverts-classification | vanilla | accuracy_score | 113.5 | 270.2 | 0 | 285.5 | 161.9 | 831.1 | 867 |
| multi-class-pred-obesity-risk | skrub-full | accuracy_score | 239.7 | 224.4 | 0 | 93.2 | 120.6 | 677.9 | 2501 |
| multi-class-pred-obesity-risk | vanilla | accuracy_score | 1545.1 | 1636.1 | 0 | 663.4 | 376.7 | 4221.3 | 10810.4 |
| reservation-cancel-classification | skrub-full | roc_auc_score | 91.8 | 114.5 | 0 | 216.8 | 222.8 | 645.9 | 1030 |
| reservation-cancel-classification | vanilla | roc_auc_score | 111.6 | 985.1 | 0 | 379.1 | 105.3 | 1581.1 | 2146.9 |
| restaurant-revenue-regression | skrub-full | root_mean_squared_error | 99.9 | 166 | 76.9 | 23.1 | 36.8 | 402.6 | 920.2 |
| restaurant-revenue-regression | vanilla | root_mean_squared_error | 50 | 241.9 | 0 | 337.7 | 170.2 | 799.7 | 867.1 |
| spaceship-titanic | skrub-full | accuracy_score | 88.9 | 52.8 | 53.8 | 24.6 | 25.1 | 245.1 | 529.2 |
| spaceship-titanic | vanilla | accuracy_score | 55.7 | 16.9 | 0 | 30.9 | 21.4 | 124.9 | 277.9 |

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

### Skrub-full extras (openai/gpt-5.4)

| Task | Metric | Tuning ran | Skill calls |
| --- | --- | --- | --- |
| abalone-regression | root_mean_squared_log_error | no | 57 |
| bike-sharing-regression | root_mean_squared_log_error | yes | 71 |
| blueberry-yield-regression | mean_absolute_error | no | 51 |
| covid19-forecasting-regression | root_mean_squared_log_error | yes | 85 |
| employee-attrition-classification | roc_auc_score | yes | 60 |
| introverts-extroverts-classification | accuracy_score | no | 91 |
| multi-class-pred-obesity-risk | accuracy_score | no | 75 |
| reservation-cancel-classification | roc_auc_score | no | 67 |
| restaurant-revenue-regression | root_mean_squared_error | yes | 60 |
| spaceship-titanic | accuracy_score | yes | 60 |

**Column guide:**
- **Task** — Benchmark task name.
- **Metric** — Validation metric for this task.
- **Tuning ran** — `yes` if the tuning stage executed for any repeat of this task.
- **Skill calls** — Total skill tool invocations (`list_skills` / `load_skill*`) in ADK logs.

## Per-run results (openai/gpt-5.4)

| Task | System | Model | Metric | Run | Best | Src val | Init | Refine | Tune | Ensemble | Gain | DataOps | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_log_error | run1 | 0.1507 | 0.1507 | 0.1513 | 0.1509 | - | 0.1507 | 0.0006 | 0.847 | 573.4 |
| abalone-regression | vanilla | openai/gpt-5.4 | root_mean_squared_log_error | run1 | 0.1506 | 0.1506 | 0.1509 | 0.1508 | - | 0.1506 | 0.0003 | 0 | 6350.8 |
| bike-sharing-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_log_error | run1 | 0.0101 | 0.0101 | 0.0433 | 0.0377 | 0.0377 | 0.0101 | 0.0333 | 0.978 | 536.8 |
| bike-sharing-regression | vanilla | openai/gpt-5.4 | root_mean_squared_log_error | run1 | 0.284 | 0.284 | 0.2969 | 0.2969 | - | 0.284 | 0.0129 | 0 | 312.7 |
| blueberry-yield-regression | skrub-full | openai/gpt-5.4 | mean_absolute_error | run1 | 337.4847 | 337.4847 | 339.8308 | 337.4847 | - | 337.4847 | 2.346 | 0.861 | 566.9 |
| blueberry-yield-regression | vanilla | openai/gpt-5.4 | mean_absolute_error | run1 | 336.6524 | 336.6524 | 337.5266 | 337.4265 | - | 336.6524 | 0.8742 | 0 | 818.1 |
| covid19-forecasting-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_log_error | run1 | 0.2965 | 0.2965 | 0.3047 | 0.2965 | 0.3078 | 0.3012 | 0.0081 | 0.857 | 1271.7 |
| covid19-forecasting-regression | vanilla | openai/gpt-5.4 | root_mean_squared_log_error | run1 | 0.0547 | 0.0547 | 0.0613 | 0.0554 | - | 0.0547 | 0.0066 | 0 | 893.2 |
| employee-attrition-classification | skrub-full | openai/gpt-5.4 | roc_auc_score | run1 | 0.8331 | 0.8331 | 0.5728 | 0.8253 | 0.8231 | 0.8331 | 0.2603 | 0.857 | 502.5 |
| employee-attrition-classification | vanilla | openai/gpt-5.4 | roc_auc_score | run1 | 0.816 | 0.816 | 0.816 | 0.816 | - | 0.8139 | 0 | 0 | 471.6 |
| introverts-extroverts-classification | skrub-full | openai/gpt-5.4 | accuracy_score | run1 | 0.9725 | 0.9725 | 0.9725 | 0.9725 | - | 0.9725 | 0 | 0.819 | 1101.6 |
| introverts-extroverts-classification | vanilla | openai/gpt-5.4 | accuracy_score | run1 | 0.9695 | 0.9695 | 0.966 | 0.9695 | - | 0.9694 | 0.0035 | 0 | 867 |
| multi-class-pred-obesity-risk | skrub-full | openai/gpt-5.4 | accuracy_score | run1 | 0.9104 | 0.9104 | 0.908 | 0.9082 | - | 0.9104 | 0.0024 | 0.75 | 2501 |
| multi-class-pred-obesity-risk | vanilla | openai/gpt-5.4 | accuracy_score | run1 | 0.9094 | 0.9094 | 0.9094 | 0.9094 | - | 0.9053 | 0 | 0 | 10810.4 |
| reservation-cancel-classification | skrub-full | openai/gpt-5.4 | roc_auc_score | run1 | 0.8981 | 0.8981 | 0.898 | 0.898 | - | 0.8981 | 0.0001 | 0.694 | 1030 |
| reservation-cancel-classification | vanilla | openai/gpt-5.4 | roc_auc_score | run1 | 0.8997 | 0.8997 | 0.8991 | 0.8995 | - | 0.8997 | 0.0006 | 0 | 2146.9 |
| restaurant-revenue-regression | skrub-full | openai/gpt-5.4 | root_mean_squared_error | run1 | 3131428.1353 | 3131428.1353 | 3161778.8129 | 3131428.1353 | 3190075.4985 | 3138027.7682 | 30350.6775 | 0.798 | 920.2 |
| restaurant-revenue-regression | vanilla | openai/gpt-5.4 | root_mean_squared_error | run1 | 2373422.6428 | 2373422.6428 | 3314702.1481 | 2401954.2177 | - | 2373422.6428 | 941279.5052 | 0 | 867.1 |
| spaceship-titanic | skrub-full | openai/gpt-5.4 | accuracy_score | run1 | 0.8171 | 0.8171 | 0.801 | 0.8148 | 0.8125 | 0.8171 | 0.0161 | 0.929 | 529.2 |
| spaceship-titanic | vanilla | openai/gpt-5.4 | accuracy_score | run1 | 0.82 | 0.82 | 0.82 | 0.82 | - | 0.8189 | 0 | 0 | 277.9 |

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

### Stage script execution — per run (openai/gpt-5.4)

| Task | System | Run | Init exec | Refine exec | Tune exec | Ensemble exec | Submit exec | Exec total | Wall total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone-regression | skrub-full | run1 | 87 | 63.3 | 0 | 98.6 | 98.3 | 347.1 | 573.4 |
| abalone-regression | vanilla | run1 | 632.5 | 1534.4 | 0 | 745.6 | 362.3 | 3274.8 | 6350.8 |
| bike-sharing-regression | skrub-full | run1 | 27.1 | 17.2 | 33.5 | 43.7 | 41.2 | 162.7 | 536.8 |
| bike-sharing-regression | vanilla | run1 | 154 | 19.7 | 0 | 19.4 | 12 | 205.1 | 312.7 |
| blueberry-yield-regression | skrub-full | run1 | 86.2 | 136.2 | 0 | 63.4 | 66.5 | 352.3 | 566.9 |
| blueberry-yield-regression | vanilla | run1 | 97.7 | 210 | 0 | 178.6 | 93.3 | 579.7 | 818.1 |
| covid19-forecasting-regression | skrub-full | run1 | 64.6 | 62.6 | 538.8 | 71.1 | 3.2 | 740.3 | 1271.7 |
| covid19-forecasting-regression | vanilla | run1 | 130.1 | 303 | 0 | 199.4 | 27.4 | 660 | 893.2 |
| employee-attrition-classification | skrub-full | run1 | 50.1 | 66.3 | 45.4 | 24.9 | 25.1 | 211.7 | 502.5 |
| employee-attrition-classification | vanilla | run1 | 55.8 | 49 | 0 | 125 | 61.3 | 291.1 | 471.6 |
| introverts-extroverts-classification | skrub-full | run1 | 72.7 | 74.3 | 0 | 124.5 | 105.9 | 377.4 | 1101.6 |
| introverts-extroverts-classification | vanilla | run1 | 113.5 | 270.2 | 0 | 285.5 | 161.9 | 831.1 | 867 |
| multi-class-pred-obesity-risk | skrub-full | run1 | 239.7 | 224.4 | 0 | 93.2 | 120.6 | 677.9 | 2501 |
| multi-class-pred-obesity-risk | vanilla | run1 | 1545.1 | 1636.1 | 0 | 663.4 | 376.7 | 4221.3 | 10810.4 |
| reservation-cancel-classification | skrub-full | run1 | 91.8 | 114.5 | 0 | 216.8 | 222.8 | 645.9 | 1030 |
| reservation-cancel-classification | vanilla | run1 | 111.6 | 985.1 | 0 | 379.1 | 105.3 | 1581.1 | 2146.9 |
| restaurant-revenue-regression | skrub-full | run1 | 99.9 | 166 | 76.9 | 23.1 | 36.8 | 402.6 | 920.2 |
| restaurant-revenue-regression | vanilla | run1 | 50 | 241.9 | 0 | 337.7 | 170.2 | 799.7 | 867.1 |
| spaceship-titanic | skrub-full | run1 | 88.9 | 52.8 | 53.8 | 24.6 | 25.1 | 245.1 | 529.2 |
| spaceship-titanic | vanilla | run1 | 55.7 | 16.9 | 0 | 30.9 | 21.4 | 124.9 | 277.9 |

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
