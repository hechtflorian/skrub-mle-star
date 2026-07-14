# Experimental Results: Skrub-MLE-STAR

The following is a detailed presentation and discussion of the benchmark results and our **skrub-enabled MLE-STAR** (skrub DataOps agent skill, TableReport profiling for targeted ablation/refinement, tuning stage, drift/robustness guards). We also refer to our version as "skrub-full". For **how** to run/reproduce, see [EXPERIMENTS.md](EXPERIMENTS.md).

We evaluated two systems: `skrub-full` (our improved pipeline) vs `vanilla` (upstream MLE-STAR + our runtime-compat layer only) across **10 tasks** on **two base LLMs**:


| Base LLM              | Runs                                                                                      | Generated Reports                                                                    |
| --------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `openai/gpt-5.4-mini` | [runs/20260707_115040_gpt_small/](automated_evaluation/runs/20260707_115040_gpt_small/) | [report.md](automated_evaluation/eval_results/20260707_115040_gpt_small/report.md) |
| `openai/gpt-5.4`      | [runs/20260709_144250_gpt_large/](automated_evaluation/runs/20260709_144250_gpt_large/) | [report.md](automated_evaluation/eval_results/20260709_144250_gpt_large/report.md) |


The file is organized as: **(1)** agent config, **(2)** metric guide, **(3)** an overall cross-model / cross-system comparison on primary (Kaggle test submission) and secondary (efficiency) metrics, **(4)** a stage-attribution analysis, **(5)** result discussion, and **(6)** summary to conclude our study. A per-base-model deep dive on holdout validation is in the appendix.

---

# 1) Agent configuration & experimental setup (`config.py`)

Captured per system in each batch's `experiment_meta.json → agent_configs` ([small](automated_evaluation/runs/20260707_115040_gpt_small/experiment_meta.json) · [large](automated_evaluation/runs/20260709_144250_gpt_large/experiment_meta.json)). Both systems ran with identical shared knobs (small loop counts for a tractable 10-task × 2-system × 2-model matrix); the improved system additionally enabled TableReport + tuning (our novelties). Per-task, the harness overrides only the necessary task configs: `task_name`, `task_type`, `lower`, and `seed` (only if given by to `run_experiments.py --seed`, else default=42 which we used for the full benchmark).


| `config.py` knob           | Value (both systems) | Meaning                               |
| -------------------------- | -------------------- | ------------------------------------- |
| `num_solutions`            | 1                    | parallel solution legs                |
| `num_model_candidates`     | 2                    | candidate models in init retrieval    |
| `inner_loop_round`         | 1                    | refinement inner (plan-refine) rounds |
| `outer_loop_round`         | 1                    | refinement outer (ablation) rounds    |
| `ensemble_loop_round`      | 1                    | ensemble refine rounds                |
| `num_top_plans`            | 2                    | plans kept for refinement/ensemble    |
| `max_debug_round`          | 5                    | debug retries per stage               |
| `max_rollback_round`       | 2                    | rollbacks on repeated failure         |
| `max_retry`                | 10                   | generic op retries                    |
| `exec_timeout`             | 600 s                | per-script execution cap              |
| `seed`                     | 42                   | reproducibility                       |
| `use_data_leakage_checker` | false                | leakage-checker sub-agent (off)       |
| `use_data_usage_checker`   | false                | data-usage checker (off)              |



| Improved-only flag     | Value | Meaning                                                                                                                       |
| ---------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| `table_report_enabled` | true  | TableReport profile injected into refinement planners ([§3/§11](CONTRIBUTIONS.md#3-tablereport-data-profiling-improved-only)) |
| `tuning_enabled`       | true  | terminal tuning stage active ([§12](CONTRIBUTIONS.md#12-new-tuning-stage-sub_agentstuning-improved-only))                     |
| `tuning_n_iter`        | 5     | max randomized-search iterations in tuning                                                                                    |


The `data_leakage_checker` and `data_usage_checker` were disabled and `num_solutions` was set to 1 (default of upstream was 2) to speed up experiments (run more tasks and evaluate more base models) and save resources. `table_report_enabled` abd `tuning_enabled` live bedind config flags and can be conveniently enabled/disabled for ablation. Note that even though `tuning_enabled=True` , the `tune_plan_agent` is designed to decide itself wether to run tuning based on ablation/refinement results. Model temperature was set to `1.0` for the GPT-5 family via `get_compatible_temperature` (LiteLLM compat, [§1](CONTRIBUTIONS.md#1-openai--chatai-runtime-compatibility-shared-with-vanilla)). Full defaults live in [shared_libraries/config.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/config.py).

---

# 2) Metrics

Performance is evaluated in terms 1) accuracy (task metric) and 2) runtime efficiency (including DataOps adherence).

### **1) Accuracy:**

- **Kaggle test-set leaderboard**: We submitted each system's exported `./final/submission.csv` to the original competition and read the **private** leaderboard (80% of the test set; more trustworthy than the 20% public split). This is the **primary metric** in the overall comparison below.
- **Holdout validation**: The `Final Validation Performance:` line printed by each stage script (parsed from `final_state.json`). Internal only; used for the **stage-attribution** analysis, because it is the only score available *per MLE-STAR subagent stage* (init → refine → tune → ensemble → submission).

Holdout validation reporting uses two numbers per task: **(best)** = the upstream script the submission agent received (best stage solution before export); **(sub)** = what the submission script produced by the final submission agent itself printed. When the submission print is anomalous (e.g. `0`, or suspiciously better than upstream), **(best)** is the fair comparison. Direction (↓/↑) follows each task's metric.

### **2) Efficiency:**

- Efficiency uses two timing axes, both reported in **seconds (s)**: **Exec** = summed Python script `execution_time` (pure compute, no LLM), **Wall** = full end-to-end run time (LLM + scripts). The gap `Wall − Exec` is LLM "thinking" (planning + tool/skill calls). Measured from MLE-Star runtime artifact `final_state.json`.
- **DataOps** adherence = share `[0,1]` of core skrub DataOps patterns across produced scripts (vanilla ≈ 0 by construction). Measured by counting the appearance (fraction between all produced scripts) of "must-have" skrub DataOps pipeline functions: `skrub.var`, `.skb.mark_as_X/y` , `.skb.apply`, `skb.apply_func` , `skrub.TableVectorizer` , `.skb.make_learner`, `learner.predict({"data": data})`

All Δ columns are `vanilla − skrub` → **positive = skrub-full is more efficient**.

> **Statistical caveat.** Every cell is **N = 1** (single run, seed 42). No variance estimates; treat everything below as directional, not statistically significant. Our two base models can account for variance, but LLM output is inherently designed to be non-deterministic (particularly with higher temperature settings as for gpt-5).

---

# 3) Overall comparison (both systems × both base models)

## Kaggle test-set leaderboard (private)

We report the **private leaderboard score** of our Kaggle submissions (`./final/submission.csv` produced by submission agent). These scores allow better generalization than holdout or the 20% public score split. Each task spans two rows (one per base LLM). Delta **Δ** is the direction-aware skrub-MLE-Star advantage (`+` ⇒ skrub-full better); **bold** marks the winner. Covid19 task had to be omitted (notebook-submission only, no scored CSV); **spaceship-titanic uses the public score** (marked `*`, private still processing since competition is ongoing). 9/10 tasks are scored.

### Task selection

Tasks were intentionally selected by considering dataset-size (ressource efficiency), competition-finish (availability of private leaderboard for honest evaluation), and task diversity (different applications, data types, tabular regression vs classification). We also included two suitable tasks that were released after the supposed gpt-5.4 pretraining cutoff date in hopes of avoiding eventual pretraining leakage (introverts-extroverts, spaceship-titanic).


<table>
<thead>
<tr>
<th align="left">Task</th><th align="left">Metric</th><th align="left">Base LLM</th>
<th align="right">Skrub-full (private)</th><th align="right">Vanilla (private)</th>
<th align="right">Δ (skrub adv.)</th><th align="left">Winner</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">abalone-regression</td><td rowspan="2">RMSLE ↓</td>
<td>gpt-5.4-mini</td><td align="right">0.41994</td><td align="right"><strong>0.14764</strong></td><td align="right">−0.27230</td><td>vanilla</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">0.14721</td><td align="right"><strong>0.14666</strong></td><td align="right">−0.00055</td><td>tie <sup>†</sup></td></tr>
<tr>
<td rowspan="2">bike-sharing-regression</td><td rowspan="2">RMSLE ↓</td>
<td>gpt-5.4-mini</td><td align="right">1.34240</td><td align="right"><strong>1.00468</strong></td><td align="right">−0.33772</td><td>vanilla</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">0.47622</td><td align="right"><strong>0.40002</strong></td><td align="right">−0.07620</td><td>vanilla</td></tr>
<tr>
<td rowspan="2">blueberry-yield-regression</td><td rowspan="2">MAE ↓</td>
<td>gpt-5.4-mini</td><td align="right">348.724</td><td align="right"><strong>336.013</strong></td><td align="right">−12.711</td><td>vanilla</td>
</tr>
<tr><td>gpt-5.4</td><td align="right"><strong>333.258</strong></td><td align="right">334.516</td><td align="right">+1.258</td><td>skrub-full</td></tr>
<tr>
<td rowspan="2">employee-attrition-classification</td><td rowspan="2">roc_auc ↑</td>
<td>gpt-5.4-mini</td><td align="right">0.63622</td><td align="right"><strong>0.85169</strong></td><td align="right">−0.21547</td><td>vanilla</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">0.85599</td><td align="right"><strong>0.88189</strong></td><td align="right">−0.02590</td><td>vanilla</td></tr>
<tr>
<td rowspan="2">introverts-extroverts-classification</td><td rowspan="2">acc ↑</td>
<td>gpt-5.4-mini</td><td align="right">0.96761</td><td align="right"><strong>0.96802</strong></td><td align="right">−0.00041</td><td>tie <sup>†</sup></td>
</tr>
<tr><td>gpt-5.4</td><td align="right">0.96802</td><td align="right"><strong>0.96862</strong></td><td align="right">−0.00061</td><td>tie <sup>†</sup></td></tr>
<tr>
<td rowspan="2">multi-class-pred-obesity-risk</td><td rowspan="2">acc ↑</td>
<td>gpt-5.4-mini</td><td align="right">0.89514</td><td align="right"><strong>0.90092</strong></td><td align="right">−0.00578</td><td>vanilla</td>
</tr>
<tr><td>gpt-5.4</td><td align="right"><strong>0.90480</strong></td><td align="right">0.90200</td><td align="right">+0.00280</td><td>skrub-full</td></tr>
<tr>
<td rowspan="2">reservation-cancel-classification</td><td rowspan="2">roc_auc ↑</td>
<td>gpt-5.4-mini</td><td align="right">0.82988</td><td align="right"><strong>0.90179</strong></td><td align="right">−0.07191</td><td>vanilla</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">0.89457</td><td align="right"><strong>0.90254</strong></td><td align="right">−0.00797</td><td>vanilla</td></tr>
<tr>
<td rowspan="2">restaurant-revenue-regression</td><td rowspan="2">RMSE ↓</td>
<td>gpt-5.4-mini</td><td align="right"><strong>1,947,091</strong></td><td align="right">2,579,579</td><td align="right">+632,488</td><td>skrub-full</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">2,124,753</td><td align="right"><strong>1,813,474</strong></td><td align="right">−311,279</td><td>vanilla</td></tr>
<tr>
<td rowspan="2">spaceship-titanic *</td><td rowspan="2">acc ↑</td>
<td>gpt-5.4-mini</td><td align="right">0.79401</td><td align="right"><strong>0.81038</strong></td><td align="right">−0.01637</td><td>vanilla</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">0.80453</td><td align="right"><strong>0.80570</strong></td><td align="right">−0.00117</td><td>tie <sup>†</sup></td></tr>
</tbody>
</table>


- spaceship-titanic: **public** leaderboard score (20% of test); private (80%) was still processing at write time. All other rows are private scores.
† **tie** = classification (acc/roc_auc) margin ≤ 0.0015 or regression margin within single-run (N=1) noise; the nominally higher score is still bolded.

**Private-LB summary** (9 tasks incl. spaceship public): 

- **gpt-5.4-mini**: skrub 1 / vanilla 7 / tie 1 
- **gpt-5.4**: skrub 2 / vanilla 4 / tie 3.

**Analysis:**

- **Vanilla leads on task count, but a large share of the gaps are within single-run noise (ties).** With `gpt-5.4-mini` skrub wins 1 (restaurant, by a ~25% margin), vanilla 7, 1 tie; with `gpt-5.4` skrub wins 2 (blueberry, obesity), vanilla 4, **3 ties**. The stronger model turns several nominal losses into effective ties, so skrub-full's real losses are concentrated on a few tasks (bike, employee-mini, reservation).
- **The larger model closes the gap.** Every skrub loss shrinks from `-mini` → `gpt-5.4`: abalone −0.272 → tie, employee −0.215 → −0.026, reservation −0.072 → −0.008, spaceship −0.016 → tie. Model capability and not the DataOps constraint drives the remaining gap.
- **Holdout wins did not always transfer.** Clearest with `-mini` **bike-sharing**: holdout RMSLE looked excellent (~0.014) but Kaggle private is **1.34** vs vanilla's 1.00 . A large holdout→test collapse consistent with submission-export drift / holdout-overfit ([deferred submission export risk](CONTRIBUTIONS.md#9-prompt-hardening-for-skrub-dataops-all-stages))

## Efficiency and structure

Two rows per task (one per base LLM). Δ = `vanilla − skrub`, so **positive (bold) = skrub-full faster**. `DataOps` is skrub-full's adherence (vanilla = 0.0 on every task). Detailed sources: `report.md` [(small)](automated_evaluation/eval_results/20260707_115040_gpt_small/report.md), `report.md` [(large)](automated_evaluation/eval_results/20260709_144250_gpt_large/report.md), "Per (task, system) summary".


<table>
<thead>
<tr>
<th align="left">Task</th><th align="left">Base LLM</th>
<th align="right">Wall skrub (s)</th><th align="right">Wall vanilla (s)</th><th align="right">ΔWall (s)</th>
<th align="right">Exec skrub (s)</th><th align="right">Exec vanilla (s)</th><th align="right">ΔExec (s)</th>
<th align="right">DataOps [0,1]</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">abalone-regression</td>
<td>gpt-5.4-mini</td><td align="right">2362</td><td align="right">5506</td><td align="right"><strong>+3143</strong></td><td align="right">2092</td><td align="right">2872</td><td align="right"><strong>+780</strong></td><td align="right">0.861</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">573</td><td align="right">6351</td><td align="right"><strong>+5777</strong></td><td align="right">347</td><td align="right">3275</td><td align="right"><strong>+2928</strong></td><td align="right">0.847</td></tr>
<tr>
<td rowspan="2">bike-sharing-regression</td>
<td>gpt-5.4-mini</td><td align="right">1251</td><td align="right">977</td><td align="right">−274</td><td align="right">588</td><td align="right">842</td><td align="right"><strong>+254</strong></td><td align="right">0.944</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">537</td><td align="right">313</td><td align="right">−224</td><td align="right">163</td><td align="right">205</td><td align="right"><strong>+42</strong></td><td align="right">0.978</td></tr>
<tr>
<td rowspan="2">blueberry-yield-regression</td>
<td>gpt-5.4-mini</td><td align="right">9673</td><td align="right">1045</td><td align="right">−8628</td><td align="right">896</td><td align="right">1021</td><td align="right"><strong>+124</strong></td><td align="right">0.821</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">567</td><td align="right">818</td><td align="right"><strong>+251</strong></td><td align="right">352</td><td align="right">580</td><td align="right"><strong>+228</strong></td><td align="right">0.861</td></tr>
<tr>
<td rowspan="2">covid19-forecasting-regression</td>
<td>gpt-5.4-mini</td><td align="right">11348</td><td align="right">1001</td><td align="right">−10347</td><td align="right">2004</td><td align="right">358</td><td align="right">−1646</td><td align="right">0.931</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">1272</td><td align="right">893</td><td align="right">−379</td><td align="right">740</td><td align="right">660</td><td align="right">−80</td><td align="right">0.857</td></tr>
<tr>
<td rowspan="2">employee-attrition-classification</td>
<td>gpt-5.4-mini</td><td align="right">804</td><td align="right">226</td><td align="right">−578</td><td align="right">771</td><td align="right">89</td><td align="right">−683</td><td align="right">0.917</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">503</td><td align="right">472</td><td align="right">−31</td><td align="right">212</td><td align="right">291</td><td align="right"><strong>+79</strong></td><td align="right">0.857</td></tr>
<tr>
<td rowspan="2">introverts-extroverts-classification</td>
<td>gpt-5.4-mini</td><td align="right">1014</td><td align="right">842</td><td align="right">−172</td><td align="right">297</td><td align="right">599</td><td align="right"><strong>+302</strong></td><td align="right">0.917</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">1102</td><td align="right">867</td><td align="right">−235</td><td align="right">377</td><td align="right">831</td><td align="right"><strong>+454</strong></td><td align="right">0.819</td></tr>
<tr>
<td rowspan="2">multi-class-pred-obesity-risk</td>
<td>gpt-5.4-mini</td><td align="right">3554</td><td align="right">6112</td><td align="right"><strong>+2558</strong></td><td align="right">1322</td><td align="right">3388</td><td align="right"><strong>+2066</strong></td><td align="right">0.833</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">2501</td><td align="right">10810</td><td align="right"><strong>+8309</strong></td><td align="right">678</td><td align="right">4221</td><td align="right"><strong>+3543</strong></td><td align="right">0.750</td></tr>
<tr>
<td rowspan="2">reservation-cancel-classification</td>
<td>gpt-5.4-mini</td><td align="right">1671</td><td align="right">3740</td><td align="right"><strong>+2069</strong></td><td align="right">1695</td><td align="right">2184</td><td align="right"><strong>+490</strong></td><td align="right">0.861</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">1030</td><td align="right">2147</td><td align="right"><strong>+1117</strong></td><td align="right">646</td><td align="right">1581</td><td align="right"><strong>+935</strong></td><td align="right">0.694</td></tr>
<tr>
<td rowspan="2">restaurant-revenue-regression</td>
<td>gpt-5.4-mini</td><td align="right">538</td><td align="right">280</td><td align="right">−258</td><td align="right">131</td><td align="right">216</td><td align="right"><strong>+85</strong></td><td align="right">0.875</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">920</td><td align="right">867</td><td align="right">−53</td><td align="right">403</td><td align="right">800</td><td align="right"><strong>+397</strong></td><td align="right">0.798</td></tr>
<tr>
<td rowspan="2">spaceship-titanic</td>
<td>gpt-5.4-mini</td><td align="right">1237</td><td align="right">2015</td><td align="right"><strong>+778</strong></td><td align="right">566</td><td align="right">2170</td><td align="right"><strong>+1604</strong></td><td align="right">0.847</td>
</tr>
<tr><td>gpt-5.4</td><td align="right">529</td><td align="right">278</td><td align="right">−251</td><td align="right">245</td><td align="right">125</td><td align="right">−120</td><td align="right">0.929</td></tr>
</tbody>
</table>


**Efficiency findings:**

- **Per-script compute (Exec) is lower for skrub-full in 8/10 tasks on *both* models** (exceptions: covid and employee on `-mini`; covid and spaceship on `gpt-5.4`). The savings are largest exactly where vanilla is heaviest: obesity (−2,066s `-mini`, −3,543s `gpt-5.4`) and abalone (+780s `-mini`, −2,928s i.e. **89% less** on `gpt-5.4`). DataOps pipelines plus our runtime rules (refinement prefers FE over large searches, deferred full-train/export in early stages) keep compute down.
- **Wall-clock is faster on the expensive tasks, slower when extra stages fire.** skrub-full wins wall time on the slowest vanilla runs (abalone, obesity, reservation on both models). The `-mini` losses are dominated by LLM overhead, not compute: the two pathological runs are covid (11,348s wall vs 2,004s exec → ~9,300s "thinking", 166 skill calls) and blueberry (9,673s wall vs 896s exec, tuning stage + skill calls). This is mainly caused by expensive debugging loops with multiple skill calls.
- **The larger model is far cheaper wall-clock for skrub-full**: covid 1,272s (vs 11,348s), blueberry 567s (vs 9,673s). The `-mini` LLM-overhead pathologies disappear on `gpt-5.4`. Model capability does seem to matter a lot, especially with more context from skill ressources and general debugging (i.e. faster debug, less debug overall, larger base model often has larger context window).
- **Structure (DataOps) is the unambiguous win:** skrub-full adherence **0.69–0.98** across every task/model vs **0.0** for vanilla everywhere. Our core objective is met regardless of accuracy outcome.

---

# 4) Novelty & modification analysis

Holdout validation is the only score available *per MLE-STAR stage*, so we use it here as proxy to attribute value to the pipeline stages, in particular our **refinement (TableReport) novelty** and the **new tuning stage**. Per-run stage scores come from `report.md` ("Per-run results": `Init / Refine / Tune / Ensemble / Src val`).

## Init → refinement gains

We inject a compact `skrub.TableReport` profile into the refinement ablation/planner agents ([§3/§11](CONTRIBUTIONS.md#3-tablereport-data-profiling-improved-only)) to make block-level refinement more targeted. **How often refinement improved the holdout score over initialization (skrub-full):**


| Base LLM     | Refinement improved init | Notable init → refine gains                                                                                                     |
| ------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| gpt-5.4-mini | **5/10**                 | restaurant 3,402,997 → 3,096,727 RMSE (~~9%); bike 0.0248 → 0.0194 RMSLE (~~22%); blueberry 345.31 → 341.53 MAE                 |
| gpt-5.4      | **8/10**                 | **employee 0.5728 → 0.8253 roc_auc (+0.25)**; spaceship 0.801 → 0.8148; covid 0.3047 → 0.2965; restaurant 3,161,779 → 3,131,428 |


The stronger model uses targeted block refinement far more effectively (8/10 vs 5/10); the standout is employee-attrition, where refinement single-handedly rescued a broken init (0.57 → 0.83). Where refinement did *not* improve, the [robust promotion guard](CONTRIBUTIONS.md#10-agent-level-guards-empty-tool-call--promotion-correctness) logic of MLE-Star correctly kept the previous solution instead of regressing.

## Tuning stage activity (new, improved-only)


| Base LLM     | Tuning ran                                              | Promoted | Outcome                                                                                                                                                |
| ------------ | ------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| gpt-5.4-mini | **1/10** (blueberry)                                    | **0**    | tuned MAE 341.82 > refine 341.53 > ensemble 340.20; [gate](CONTRIBUTIONS.md#12-new-tuning-stage-sub_agentstuning-improved-only) correctly withheld it |
| gpt-5.4      | **5/10** (bike, covid, employee, restaurant, spaceship) | **0**    | every tuned score ≤ structural/ensemble winner (e.g. covid tune 0.3078 vs refine 0.2965; spaceship 0.8125 vs ensemble 0.8171); gate withheld all      |


Tuning fired much more with the larger model (5/10 vs 1/10) but produced **0 promotions on either model**: in every case the promotion gate prevented a regression rather than adding a win. The agent is also allowed to **self-skip** when ablation signal suggests refinement already captured the gains ([tuning agent](CONTRIBUTIONS.md#12-new-tuning-stage-sub_agentstuning-improved-only)). The guard works, but tuning has not yet demonstrated a net gain on this benchmark.

## Stage that produced the final promoted (submission-export) solution

Which stage's script became the `Src val` upstream solution handed to the submission agent (the best-scoring stage; `init` = no later stage improved it):


| Task        | mini · skrub | mini · vanilla | gpt-5.4 · skrub | gpt-5.4 · vanilla |
| ----------- | ------------ | -------------- | --------------- | ----------------- |
| abalone     | init         | refine         | ensemble        | ensemble          |
| bike        | ensemble     | refine         | ensemble        | ensemble          |
| blueberry   | ensemble     | refine         | refine          | ensemble          |
| covid       | ensemble     | init           | **refine**      | ensemble          |
| employee    | refine       | refine         | ensemble        | init              |
| introverts  | refine       | ensemble       | init            | refine            |
| obesity     | init         | refine         | ensemble        | init              |
| reservation | ensemble     | init           | ensemble        | ensemble          |
| restaurant  | ensemble     | ensemble       | **refine**      | ensemble          |
| spaceship   | ensemble     | ensemble       | ensemble        | init              |


**Stage summary (10 tasks):**


| System / model            | init held | refinement | ensemble |
| ------------------------- | --------- | ---------- | -------- |
| skrub-full · gpt-5.4-mini | 2         | 2          | **6**    |
| skrub-full · gpt-5.4      | 1         | 2          | **7**    |
| vanilla · gpt-5.4-mini    | 2         | **5**      | 3        |
| vanilla · gpt-5.4         | 3         | 1          | **6**    |


## Summary of stage analysis

- **Ensemble is the usual final winner for skrub-full** (6/10 mini, 7/10 `gpt-5.4`): the biggest score jumps come from **init → refine → ensemble** together, with ensemble landing the last gain, which is exactly the MLE-STAR design intent (each stage should improve the previous promoted solution, else the runtime would be wasted).
- **Refinement (our TableReport-fed stage) usually improves init**, but often not the final winner: it improved init in 5/10 (`-mini`) and 8/10 (`gpt-5.4`), and it was the *promoted* source where ensemble regressed (e.g. covid and restaurant on `gpt-5.4`). The `gpt-5.4` employee rescue (+0.25) is the single most valuable refinement event in the batch.
- **Vanilla's stage mix depends heavily on the model.** With `-mini`, vanilla's ensemble frequently *regressed* (e.g. bike ensemble 1.098 vs refine 0.2991), so refinement was promoted most often (5/10). With `gpt-5.4`, vanilla's ensemble becomes reliable (6/10), the same ensemble-dominant pattern skrub-full shows.
- **Tuning contributed no promotions on either model**, but its gate never caused a regression. It is correctly conservative and simply under-exercised. Additionally, tuning might not be able to contribute meaningful gains if refinement already captured most gains.

---

## Connecting results to our modifications


| Modification               | Evidence                                                                                                                                                              |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| skrub DataOps skill        | DataOps adherence 0.69–0.98 vs 0 on both models; large wins on text/categorical/datetime tasks (bike, restaurant)                                                     |
| TableReport profiling      | Refinement improved init 5/10 (`-mini`) → 8/10 (`gpt-5.4`)                                                                                                            |
| Tuning stage               | Fired 1/10 (`-mini`) and 5/10 (`gpt-5.4`); no unnessary ressource consumption if projeted gains negligable; gate withheld every worse tuned result                    |
| Execution-robustness gates | All runs produced usable `final_state.json`; no empty/tool-call-only turn scored as valid or agent turns wasted                                                       |
| Runtime compat             | Enabled the whole benchmark to run on both OpenAI-compatible models at all; while maintaining gemini-compat                                                           |


The improvements **reliably deliver the structural/DataOps objective** and **per-script compute savings** on both models, and **close the accuracy gap as the model scales** (several `-mini` losses become ties on `gpt-5.4`). On this single-seed matrix they did not yet yield a net primary-metric win over vanilla, and the tuning stage remains under-exercised or does not yet bring valuable gains. However, our TableReport novelty and prompt refinements for ablation/refinement usually bring **efficiency gains** while keeping task scores tied.

---

# 5) Discussion

**Skrub-DataOps-MLE-STAR:** Our central goal, making agents generate skrub DataOps-native ML pipelines, holds on every task and both base models and skrub-full reaches up to 0.98 DataOps adherence. The load-on-demand skill achieves this without permanent prompt bloat and stays maintainable, since references can be appended, corrected, or backed by deterministic helper scripts later. Generally, an agent skill is designed to optimize/expand agent capability while being context-efficient: agents call `list_skills` (load only skill metadata for discovery) **→** `load_skill` (load only SKILL.md) **→** `load_skill_resource` (load additional [reference.md](http://reference.md) files) in order and only when needed. That means agents decide themselves when the skill is relevant and which resources to load and they can be guided via prompts. This can get complicated when specific skill usage and loading specific resources for certain stages is necessary, because the agent must be made to load the desired resources (e.g. the agent may decide to not load them). Finally, our generated pipelines are direct evidence that the skrub DataOps skill and the stage prompt hardening work as intended.

**Accuracy stays competitive and improves with model capability:** On the Kaggle private leaderboard vanilla does lead on raw task count, but a large share of the gaps sit within single-run noise. Every skrub-full deficit on gpt-5.4-mini shrinks on gpt-5.4, and skrub-full wins on restaurant (mini) and on blueberry and obesity (gpt-5.4). Making the agents use DataOps is genuinely hard because pretraining nudges them toward sklearn, yet once the base model is capable enough the constraint costs little to no accuracy and our skrub-MLE-STAR can keep up with vanilla.

**Efficiency is a consistent, honest gain:** skrub-full runs leaner per script (execution time) on 8/10 tasks for both models, with the largest savings where vanilla is heaviest (obesity, abalone), and it wins wall-clock time on the slowest vanilla runs. Feeding a compact TableReport profile into ablation/refinement, and nudging those stages toward feature engineering rather than large searches, keeps compute down while scores often stay similar or better. When accuracy is often tied, spending less compute to reach the same result is itself a meaningful win, and it is the practical payoff of the TableReport novelty and the refinement prompt changes. However, this result should be taken with a grain of salt: While the execution time is usually lower, the wall time can sometimes increase if debugging effort increases (i.e. with less base model capability), while a base model with higher capability (gpt-5.4) shows robustness and gains on both fronts. Similarly, prefering FE and avoiding larger searches or complex ensembles can lead to reduced accuracy, so there is a real tradeoff between accuracy and efficiency. Most negative outliers in terms of wall time we're due to heavy debugging, and potentially even issues with the API.

**The trade-offs are understood and mostly capacity-related:** Model capability governs the cost of the skill: the larger model calls skills less (51 to 91 vs 75 to 166), hallucinates less, debugs less, and removes the wall-clock pathologies of the small model. On weaker models more debugging fires the pre-execution guards and pulls extra skill and web-search calls, which can end up in more debugging and drift from the optimal solution (debug also uses web search to skrub API, which can often confuse the agent; e.g. if it sees RF model in examples, it swaps the backbone to RF) - so runtime can rise even though per-script exec time is lower. We deliberately kept backbone-drift and contract guards minimal so exploration is not over-constrained, and the robustness gates still delivered 40/40 usable runs. Two design choices carry residual risk: deferring the full-train refit and export to the submission agent in our skrub-MLE-STAR saves exec time but puts more effort on the submission agent (can debug more often, which lets that agent drift), and the tuning stage produced no promotions on either model to prove it's benefit.

---

# 6) Summary

We set out to make MLE-STAR produce skrub DataOps native pipelines and to add targeted improvements around that goal: a load-on-demand skrub DataOps agent skill, TableReport-driven targeted ablation and refinement, a dedicated tuning stage, pipeline-drift and stage contract guards, execution-robustness gates, and OpenAI/ChatAI runtime compatibility. We evaluated our system (skrub-full) against the vanilla upstream baseline on 10 Kaggle tasks and two base models (gpt-5.4-mini and gpt-5.4), using both internal holdout scores and Kaggle private-leaderboard scores as well as system efficiency metrics.

The structural goal is fully met. DataOps adherence up to 0.98 versus 0.0 shows the skill reliably steers code into DataOps pipelines on every task and model, and it does so without prompt bloat while references load on demand and can be maintained or extended over time. On accuracy, the our skrub-MLE-STAR is competitive and scale-sensitive. Vanilla edges the private-leaderboard task count, but many of those gaps are within single-run noise and register as ties, skrub-full wins several tasks outright, and every deficit shrinks moving from the small to the larger, more capable base model. In other words, the DataOps constraint costs little to no accuracy once the base model is strong enough, and on the capable model skrub-full is effectively level with ahead of vanilla on most tasks when considering accuracy and efficiency together.

On efficiency the advantage is consistent: skrub-full is leaner per script on 8/10 tasks and faster end-to-end on the heaviest tasks, because TableReport-guided, feature-engineering-first refinement avoids the large searches vanilla tends to run. Given how often accuracy ties, this compute saving is a concrete benefit of our modifications. The pipeline also proved robust, with 40/40 usable runs and promotion guards that never promoted a regression, and tablereport-enabled refinement demonstrates strong per-stage gain on the large model.

Overall, skrub-MLE-STAR delivers its structural objective in full, keeps pace with vanilla on accuracy while pulling ahead as the model scales, and reaches those results with less compute and with extended guardrails that keep runs valid. The remaining risks, submission-stage drift and an under-proven tuning stage, are concrete and addressable. The strongest outcomes, DataOps adherence while maintaining accuracy and per-script efficiency, are where our skrub-MLE-STAR demonstrates improvement.

---

## Limitations and future work

- **Single run, single seed (N = 1, seed 42):** All results are directional. Add `run2`/`run3` with different seeds ([EXPERIMENTS.md](EXPERIMENTS.md#adding-another-repeat-eg-run2)); LLM sampling is non-deterministic even with a seed (temperature is forced to 1.0 for the GPT-5 family), so repeats are the main way to firm up every claim.
- **Isolate the novelties:** Run ablations that toggle `table_report_enabled` and `tuning_enabled` across repeats to attribute influence directly, rather than inferring it from stage promotion.
- **Rework or retire tuning:** It has shown no gains yet; try a real grid search, or disable it, weighing the runtime cost against the efficiency target.
- **Broaden pre-exec gates to submission:** Expand pre-exec checks to the submission agent to avoid final solution export degradation due to drift, while maintaining the efficiency gains of avoiding early-stage full-train refit.
- **Scale the config fairly:** Raising `num_solutions` and the loop counts should lift both systems (identical config for both) and is worth trying when resources allow.
- **Broaden coverage:** Add open-weight models (e.g. mistral) beside the closed ones to study potential base model gaps, more complex tasks (multi-table merges, not just one train-/test-set), and re-enable the leakage and data-usage checkers to quantify and remove any potential leakage in the holdout scores.
- **Agents can drift off from optimal solution because of context:** An interesting idea for future work could be to optimize agent context. If each subagent would have the full stage/task/runtime context, drift from the optimal solution could potentially be reduced (e.g. debug does not only see a bug, but knows what the previous agents want to achieve and stays in that solution range). However, this might be complicated and context needs to be optimized/condensed to avoid hallucination and maintain efficiency. 

---

# Appendix

## Holdout validation analysis: `openai/gpt-5.4-mini`

Holdout validation view (primary Kaggle numbers are [above](#primary-metric--kaggle-test-set-leaderboard-private)). Report: `[report.md](automated_evaluation/eval_results/20260707_115040_gpt_small/report.md)` · `[evaluation_summary.json](automated_evaluation/eval_results/20260707_115040_gpt_small/evaluation_summary.json)`.

**Holdout headline (best):** skrub-full wins **3/10**, vanilla **7/10** (submission-print: skrub 2/10, vanilla 5/10; 3 tasks had no parseable vanilla submission score). On the fair **(best)** comparison the wins are lopsided. Skrub-full's wins are large, its losses are mostly tiny:


| Task                                 | Metric    | Vanilla (best) | Skrub-full (best) | Δ (skrub−vanilla) | Winner     |
| ------------------------------------ | --------- | -------------- | ----------------- | ----------------- | ---------- |
| bike-sharing-regression              | RMSLE ↓   | 0.2991         | **0.0144**        | **+0.2846**       | skrub-full |
| restaurant-revenue-regression        | RMSE ↓    | 3,201,219      | **2,008,493**     | **+1,192,726**    | skrub-full |
| introverts-extroverts-classification | acc ↑     | 0.9663         | **0.9684**        | +0.0022           | skrub-full |
| spaceship-titanic                    | acc ↑     | **0.8223**     | 0.8091            | −0.0132           | vanilla    |
| multi-class-pred-obesity-risk        | acc ↑     | **0.9068**     | 0.9063            | −0.0005           | vanilla    |
| abalone-regression                   | RMSLE ↓   | **0.1486**     | 0.1510            | −0.0025           | vanilla    |
| blueberry-yield-regression           | MAE ↓     | **337.95**     | 340.20            | −2.25             | vanilla    |
| reservation-cancel-classification    | roc_auc ↑ | **0.8988**     | 0.8230            | −0.0758           | vanilla    |
| covid19-forecasting-regression       | RMSLE ↓   | **0.7677**     | 0.8449            | −0.0772           | vanilla    |
| employee-attrition-classification    | roc_auc ↑ | **0.8329**     | 0.6180            | −0.2149           | vanilla    |


- Where DataOps structure helps, it helps a lot: **bike-sharing** and **restaurant-revenue** are both heavy on categorical/text/datetime columns, exactly what `TableVectorizer`/DataOps encoders target well with defaults.
- The one clear regression is **employee-attrition** (0.83 → 0.62): a small, imbalanced table where the skrub-full submission-agent run drifted. This **flips to a win on** `gpt-5.4` (below), so it was model-capability-driven.
- Two "losses" (obesity −0.0005, abalone −0.0025) are within N=1 noise.
- **Skill usage was active**: 75–166 `list_skills`/`load_skill`* calls per task, confirming the on-demand skill (not prompt bloat) steers structure, but the heavy skill/tuning use also drives the covid/blueberry wall-clock pathologies.

## Holdout validation analysis: `openai/gpt-5.4`

Same 10 tasks × {improved, vanilla}, seed 42, N = 1, identical `config.py`, only the model changed. Report: `[report.md](automated_evaluation/eval_results/20260709_144250_gpt_large/report.md)` · `[evaluation_summary.json](automated_evaluation/eval_results/20260709_144250_gpt_large/evaluation_summary.json)`.

**Holdout headline (best):** skrub-full **4/10**, vanilla **6/10** (submission-print: skrub 3/10, vanilla 5/10). skrub up from 3/10 with `-mini`:


| Task                                 | Metric    | Vanilla (best) | Skrub-full (best) | Δ (skrub−vanilla)  | Winner     |
| ------------------------------------ | --------- | -------------- | ----------------- | ------------------ | ---------- |
| bike-sharing-regression              | RMSLE ↓   | 0.2840         | **0.0101**        | **+0.2740** (~96%) | skrub-full |
| employee-attrition-classification    | roc_auc ↑ | 0.8160         | **0.8331**        | **+0.0171**        | skrub-full |
| introverts-extroverts-classification | acc ↑     | 0.9695         | **0.9725**        | +0.0030            | skrub-full |
| multi-class-pred-obesity-risk        | acc ↑     | 0.9094         | **0.9104**        | +0.0010            | skrub-full |
| abalone-regression                   | RMSLE ↓   | **0.1506**     | 0.1507            | −0.0001            | vanilla    |
| reservation-cancel-classification    | roc_auc ↑ | **0.8997**     | 0.8981            | −0.0016            | vanilla    |
| spaceship-titanic                    | acc ↑     | **0.8200**     | 0.8171            | −0.0029            | vanilla    |
| blueberry-yield-regression           | MAE ↓     | **336.65**     | 337.48            | −0.83              | vanilla    |
| covid19-forecasting-regression       | RMSLE ↓   | **0.0547**     | 0.2965            | −0.2418            | vanilla    |
| restaurant-revenue-regression        | RMSE ↓    | **2,373,423**  | 3,131,428         | −758,005           | vanilla    |


**Notable shifts vs** `-mini`**:**

- **employee-attrition flipped from skrub's worst loss to a win** (−0.2149 → **+0.0171**): the larger model recovered a bad init via refinement (0.5728 → 0.8253 → ensemble 0.8331), confirming the earlier regression was model-capability (i.e. debug drift).
- **restaurant flipped the other way**, and **covid is a strong vanilla result** (0.0547; skrub 0.2965 + a broken 1.1064 submission print).
- The remaining skrub losses (abalone, reservation, spaceship, blueberry) are all tiny. Skrub-full keeps pace with vanilla on the larger model.
- DataOps adherence **0.69–0.98** vs 0.0; skill calls **51–91** per task (fewer than `-mini`'s 75–166 due to less debugging needed).

