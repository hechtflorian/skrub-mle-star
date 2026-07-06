# MLE-STAR automated evaluation report

- Generated (UTC): 2026-07-05T19:17:53.378006+00:00
- Git SHA: `bb96a9c`
- Runs root: `automated_evaluation/runs/20260705_191748`
- Seed: `42`
- Model: `openai/gpt-5.4-mini`

## Scanned tasks

Complete: **10 / 10**

| Task | Type | Metric | Ready |
| --- | --- | --- | --- |
| abalone-regression | Tabular Regression | root_mean_squared_log_error | yes |
| bank-dataset-classification | Tabular Classification | roc_auc_score | yes |
| blueberry-yield-regression | Tabular Regression | mean_absolute_error | yes |
| calories-burned-regression | Tabular Regression | root_mean_squared_log_error | yes |
| covid19-forecasting-regression | Tabular Regression | root_mean_squared_log_error | yes |
| diabetes-classification | Tabular Classification | roc_auc_score | yes |
| introverts-extroverts-classification | Tabular Classification | accuracy_score | yes |
| multi-class-pred-obesity-risk | Tabular Classification | accuracy_score | yes |
| podcast-listening-hours-regression | Tabular Regression | root_mean_squared_error | yes |
| spaceship-titanic | Tabular Classification | accuracy_score | yes |

**Column guide:**
- **Task** — Task folder name under `tasks/`.
- **Type** — Problem type (regression or classification).
- **Metric** — Primary metric named in the task pack (e.g. RMSE, accuracy).
- **Ready** — `yes` if `train.csv`, `test.csv`, and `task_description.txt` exist.

## Archived runs analyzed

Total runs with usable `final_state.json`: **0**
