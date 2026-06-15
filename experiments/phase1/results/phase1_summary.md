# Phase 1 summary

**Grid:** 2 tasks × 2 systems × 3 repeats · `gpt-5.4-mini` · seed 42  
**Hypothesis (operational):** skrub DataOps makes the pipeline **easier to run** — higher DataOps adherence, **shorter Python execution**, **fewer debug rounds** — not necessarily better holdout scores (although we hope when refinement/tuning fully stabilized).

**Excluded as probable leakage (housing only):**


| Run               | Reason                                                            |
| ----------------- | ----------------------------------------------------------------- |
| skrub-full `run1` | Ensemble/submission RMSE ~42k (optimistic vs ~52–56k honest band) |
| vanilla `run2`    | Refinement/submission RMSE ~10k (full-train → eval holdout)       |


All **Spaceship Titanic** runs kept (no confirmed leakage). Housing aggregates below use **n=2** per system; Titanic uses **n=3**.

### Timing metrics (from `analyze_run.py` + archived artifacts)


| Metric                          | Source                                            | Meaning                                                                                                                                                |
| ------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Val score**                   | `*_exec_result_*` → `score` in `final_state.json` | Holdout metric printed as `Final Validation Performance`                                                                                               |
| **Python exec (stage / total)** | Same keys → `execution_time`                      | Wall time of each `subprocess.run(python …)`; **summed** per stage across all **scored** script runs (includes retries, ablation, inner-loop attempts) |
| **Wall time (run total)**       | ADK log `Script started on` → `Script done on`    | Full session elapsed time: LLM calls, tools/skills, waiting, **plus** all Python exec                                                                  |
| **Python fraction**             | `exec_seconds / wall_seconds`                     | Share of the run spent in Python subprocesses vs agent overhead                                                                                        |
| **Exec runs (stage)**           | Count of scored exec results per stage            | How many Python scripts ran (proxy for retries / stage churn)                                                                                          |
| **Debug rounds**                | ADK log lines from `bug_summary` / `debug_agent`  | LLM debug iterations per stage                                                                                                                         |


Positive `gain_refinement` / `gain_tuning` = improvement in metric direction (lower RMSE or higher accuracy).

---

## California housing (RMSE ↓, n=2 per system after exclusions)

### Holdout val score by stage (mean ± std)


| Stage                  | vanilla        | skrub-full     |
| ---------------------- | -------------- | -------------- |
| Init (`train0`)        | 54 744 ± 0     | 55 293 ± 222   |
| Refine (best promoted) | 54 650 ± 133   | 52 760 ± 40    |
| Tune search            | —              | 54 809 ± 1 320 |
| Tune bake              | —              | 54 809 ± 1 320 |
| Ensemble (best)        | 52 279 ± 3 089 | 52 597 ± 203   |
| Submission             | 52 568 ± 3 498 | 52 597 ± 203   |


Scores sit in the same **~52–56k** band; refinement helped skrub on run2–3. Tuning did not beat structural (search ≈ bake ≈ structural path).

### Python exec time by stage (mean ± std, seconds)


| Stage          | vanilla         | skrub-full    |
| -------------- | --------------- | ------------- |
| Initialization | 108 ± 44        | 113 ± 86      |
| Refinement     | 443 ± 211       | 68 ± 38       |
| Tuning         | —               | 143 ± 81      |
| Ensemble       | 441 ± 450       | 88 ± 50       |
| Submission     | 229 ± 294       | 51 ± 12       |
| **Total exec** | **1 221 ± 577** | **462 ± 267** |


Skrub **~2.6× less** total Python execution time on the two clean housing runs.

### Run timing (wall vs Python exec)


|                                      | vanilla       | skrub-full  |
| ------------------------------------ | ------------- | ----------- |
| **Wall time** (full ADK session)     | 1 844 ± 29 s  | 786 ± 132 s |
| **Python exec** (total, scored runs) | 1 221 ± 577 s | 462 ± 267 s |
| **Python fraction** (exec / wall)    | 66% ± 30%     | 57% ± 24%   |


Wall time is the practical “how long did the run take”; skrub sessions were **~2.3× shorter** on wall as well as exec.

### Scored Python runs per stage (mean ± std)


| Stage          | vanilla | skrub-full |
| -------------- | ------- | ---------- |
| Initialization | 4 ± 0   | 4 ± 0      |
| Refinement     | 4 ± 0   | 4 ± 0      |
| Tuning         | —       | 2 ± 0      |
| Ensemble       | 2 ± 3   | 2 ± 0      |
| Submission     | 1 ± 0   | 1 ± 0      |


Same script counts per stage; skrub runs each script faster, not fewer attempts (on these two runs).

### Refinement promotion & tuning


|                                | vanilla                                               | skrub-full                                                              |
| ------------------------------ | ----------------------------------------------------- | ----------------------------------------------------------------------- |
| **Refine promoted over init?** | run1: no (`gain_refinement` 0); run3: yes (+189 RMSE) | run2–3: yes (+2.4k / +2.4k RMSE)                                        |
| `**tune_winner_source`**       | — (no tuning stage)                                   | **structural** on both runs                                             |
| **Tuning beat structural?**    | —                                                     | **No** (`gain_tuning` negative: search/bake worse than promoted refine) |


### Debug rounds (bug_summary / debug_agent lines in log)


| Stage          | vanilla    | skrub-full  |
| -------------- | ---------- | ----------- |
| Initialization | 8 ± 0      | 3 ± 1       |
| Refinement     | 4 ± 0      | 12 ± 9      |
| Tuning         | 0          | 0           |
| Ensemble       | 2 ± 3      | 0           |
| **Total**      | **14 ± 3** | **15 ± 10** |


Refinement debug is **higher** on skrub despite faster exec. DataOps adherence: **0.00** (vanilla) vs **0.71** (skrub).

---

## Spaceship Titanic (accuracy ↑, n=3)

### Holdout val score by stage (mean ± std)


| Stage                  | vanilla         | skrub-full       |
| ---------------------- | --------------- | ---------------- |
| Init                   | 0.8244 ± 0.0012 | 0.7888 ± 0.0027  |
| Refine (best promoted) | 0.8244 ± 0.0012 | 0.7888 ± 0.0027  |
| Tune search            | —               | 0.7700 ± 0.0325* |
| Tune bake              | —               | 0.7851 ± 0.0018  |
| Ensemble (best)        | 0.8171 ± 0.0074 | 0.7880 ± 0.0026  |
| Submission             | 0.8126 ± 0.0006 | 0.7872 ± 0.0012  |


 Tune search used wrong metric on early runs (RMSE-scale ~~0.47–0.79); treat as diagnostic only. Vanilla ahead by **~~2.5 pp** on init/refine/submission.

### Python exec time by stage (mean ± std, seconds)


| Stage          | vanilla         | skrub-full      |
| -------------- | --------------- | --------------- |
| Initialization | 699 ± 547       | 386 ± 81        |
| Refinement     | 975 ± 577       | 271 ± 92        |
| Tuning         | —               | 263 ± 309       |
| Ensemble       | 494 ± 270       | 227 ± 93        |
| Submission     | 74 ± 36         | 152 ± 84        |
| **Total exec** | **2 243 ± 824** | **1 298 ± 454** |


Skrub **~42% less** total Python execution time (high variance on vanilla run3).

### Run timing (wall vs Python exec)


|                                      | vanilla       | skrub-full    |
| ------------------------------------ | ------------- | ------------- |
| **Wall time** (full ADK session)     | 2 905 ± 543 s | 2 388 ± 504 s |
| **Python exec** (total, scored runs) | 2 243 ± 824 s | 1 298 ± 454 s |
| **Python fraction** (exec / wall)    | 76% ± 13%     | 55% ± 18%     |


Skrub used less wall and exec time; a **larger share** of vanilla wall time is Python (long CatBoost trains), while skrub wall time is more **LLM-heavy** (lower Python fraction).

### Scored Python runs per stage (mean ± std)


| Stage          | vanilla | skrub-full |
| -------------- | ------- | ---------- |
| Initialization | 4 ± 0   | 4 ± 0      |
| Refinement     | 4 ± 0   | 4 ± 0      |
| Tuning         | —       | 1.7 ± 1.2  |
| Ensemble       | 2 ± 0   | 2 ± 0      |
| Submission     | 1 ± 0   | 1 ± 0      |


### Refinement promotion & tuning


|                                | vanilla                                    | skrub-full                                 |
| ------------------------------ | ------------------------------------------ | ------------------------------------------ |
| **Refine promoted over init?** | **No** on all 3 runs (`gain_refinement` 0) | **No** on all 3 runs (`gain_refinement` 0) |
| `**tune_winner_source`**       | —                                          | **structural** on all 3 runs               |
| **Tuning beat structural?**    | —                                          | **No** (`gain_tuning` ≤ 0 every run)       |


### Debug rounds


| Stage          | vanilla    | skrub-full |
| -------------- | ---------- | ---------- |
| Initialization | 7 ± 7      | 5 ± 2      |
| Refinement     | 0 ± 0      | 12 ± 4     |
| Tuning         | 0 ± 0      | 1 ± 2      |
| Ensemble       | 4 ± 3      | 0 ± 0      |
| **Total**      | **11 ± 7** | **18 ± 5** |


Skrub had **more** refinement (and slightly more total) debug lines; vanilla spent debug budget on init/ensemble instead. DataOps adherence: **0.00** vs **0.64**.

---

## Per-run reference (included runs only)

**Housing — val scores (init → refine → ensemble → submission)**


| run  | vanilla                           | skrub-full                        |
| ---- | --------------------------------- | --------------------------------- |
| run1 | 54 744 → 54 744 → 50 095 → 50 095 | *excluded*                        |
| run2 | *excluded*                        | 55 450 → 52 788 → 52 740 → 52 740 |
| run3 | 54 744 → 54 556 → 54 462 → 55 041 | 55 136 → 52 732 → 52 454 → 52 454 |


**Titanic — val scores (init → refine → submission)**


| run  | vanilla               | skrub-full            |
| ---- | --------------------- | --------------------- |
| run1 | 0.824 → 0.824 → 0.813 | 0.787 → 0.787 → 0.788 |
| run2 | 0.826 → 0.826 → 0.813 | 0.787 → 0.787 → 0.786 |
| run3 | 0.823 → 0.823 → 0.812 | 0.792 → 0.792 → 0.787 |


---

## Conclusion

We did **not** expect large score gains; holdout quality was a hope, not the main hypothesis.


| Hypothesis signal                | Housing (n=2)                                      | Titanic (n=3)                                                                                  |
| -------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Faster Python exec**           | **Yes** (~2.6×)                                    | **Yes** (~42%)                                                                                 |
| **Shorter wall time (full run)** | **Yes** (~2.3×)                                    | **Yes** (~18%)                                                                                 |
| **High DataOps adherence**       | **Yes** (~0.71)                                    | **Yes** (~0.64)                                                                                |
| **Tuning beat structural**       | —                                                  | **No** (`tune_winner_source=structural` all runs; stage needs refinement or could be disabled) |
| **Refine promoted over init**    | run3 only (housing)                                | housing run2–3 only; Titanic none                                                              |
| **Fewer debug rounds**           | **No** (similar total; more refine debug on skrub) | **No** (skrub +7 total vs vanilla)                                                             |
| **Better val scores**            | **Rough tie** (~52–56k RMSE)                       | **No** (vanilla ~+2.5 pp accuracy)                                                             |


**Takeaway:** skrub-full delivered the **operational** part of the bet — structured DataOps code with ** materially lower exec time** on both tasks — but **did not** reduce debug churn (refinement/debug drift still costly on skrub). Scores were comparable on clean housing runs; vanilla won on Titanic. Post-phase fixes (metric generalization, `tune_best_params`, planned backbone gate) target the remaining debug and tuning noise.

---

## Artifacts


| File                                                     | Description                            |
| -------------------------------------------------------- | -------------------------------------- |
| `[phase1_all_runs_12.csv](phase1_all_runs_12.csv)`       | All 12 runs (`analyze_run` row fields) |
| `[phase1_by_task_system.csv](phase1_by_task_system.csv)` | Means over full 12-run grid            |
| `[../manifest.csv](../manifest.csv)`                     | Run index + archive paths              |


Regenerate: `python test-scripts/aggregate_runs.py --root experiments/phase1 --out experiments/phase1/results/phase1_all_runs.csv`

Legacy `spaceship-titanic/skrub-full/.../legacy/run1` excluded from tables above.

---

## TL;DR

**Setup:** 2 tasks (California housing, Spaceship Titanic) × 2 systems × 3 repeats · `gpt-5.4-mini` · seed 42. Compare **vanilla MLE-STAR** (`mle-star_vanilla`) vs **skrub-full** (`mle-star_improved`: DataOps skills, **TableReport** in refinement ablation, **terminal tuning** with in-graph `choose_`* → search → bake).

**Hypothesis:** skrub makes tabular pipelines **easier to run** (DataOps adherence, less Python/wall time, fewer debugs) — **not** necessarily better holdout scores, kinda expected (although we hoped, but maybe we get more stable results on scaled eval).

### Vanilla vs skrub-full


|                          | Vanilla                                               | Skrub-full                                                                                      |
| ------------------------ | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Scores (honest runs)** | Housing ~~52–55k RMSE; Titanic **~~0.813** accuracy   | Housing ~~52–53k RMSE (tie); Titanic **~~0.787** (−2.5 pp)                                      |
| **Python exec**          | Slower (housing ~1 221 s; Titanic ~2 243 s)           | **Faster** (~2.6× housing, ~42% Titanic)                                                        |
| **Wall time (full run)** | Longer (1 844 s / 2 905 s)                            | **Shorter** (~2.3× / ~18%)                                                                      |
| **DataOps adherence**    | ~0 (expected)                                         | **~0.64–0.71** — pipelines mostly DataOps-native                                                |
| **Debug rounds**         | Fewer on refine (Titanic 0); similar total on housing | **More refinement debug** (backbone, preprocessing drift); tuning debug on Titanic run1         |
| **Tuning stage**         | None                                                  | Runs every time; `**tune_winner_source=structural` always** — search never beat promoted refine |
| **Refine promotion**     | Housing run3 only (+189 RMSE); Titanic none           | Housing run2–3 (+2.4k RMSE); Titanic none                                                       |


Exclude **housing skrub run1** and **vanilla run2** from score claims (probable leakage). Titanic runs kept as-is.

### How our additions performed

- **TableReport (refinement):** Present on all skrub runs (~1.2k chars profile → ablation agent). Helped steer structural ablations on housing (refine promoted on run2–3); on Titanic, ablation ran but **no score gain** over init — profile useful, promotion logic still weak (drift, wrong tune metrics).
- **Terminal tuning:** **Mechanically works** (search + bake complete; params mapped to state). **Did not improve** final holdout: structural solution won every run. Titanic tune search sometimes used **wrong metric** (RMSE-style scores for classification - should be fixed now); housing tune ran but `gain_tuning` negative. Tuning is **optional/disabled candidate** until drift (e.g. backbone) fixes land. Additionally, tuning only focuses on model hyperparams: maybe we should steer away and more to feature engineering search.
- **DataOps pipeline:** **Works reasonably well** (~65–70% adherence) — `skrub.var`, `.skb.apply`, holdout bind patterns appear in generated code; main gaps were debug drift (e.g. model swaps, preprocessing changes), not absence of DataOps.

### Benefits of adding skrub (what we got)

- **Faster runs:** Much less Python time per stage (especially refinement/ensemble on housing); materially shorter wall clock on housing.
- **Structured tabular code:** Consistent DataOps graph instead of ad-hoc sklearn scripts; easier to audit holdout binding and FE in one pipeline.
- **Richer refinement loop:** TableReport + skrub skills enable targeted ablation/tuning plans (even when scores did not win in our first simple experiments; maybe needs more robustness (e.g. maybe add tablereport summary agent instead of just compact json)).
- **Operational hypothesis largely confirmed** on speed + structure; **not confirmed** on fewer debugs or better accuracy.

### Bottom line

Skrub-full is a **credible operational upgrade** (speed, DataOps structure, refinement tooling) but **not a score win** vs vanilla in phase 1. Vanilla still wins Titanic; housing is a tie on clean runs. **Tuning and TableReport are implemented and active** but need hardening (backbone gate, `tune_best_params`, deterministic drift checks, steering more towards feature encoder tuning, etc.) before they add value over structural refine. Next step: fixes already in flight, then selective re-runs — not a full 12-run grid until tuning/refine stabilize.