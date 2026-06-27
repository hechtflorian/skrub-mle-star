# Status Report: skrub-enabled MLE-STAR

**Date:** June 2026  
**Branch / snapshot:** `improve-refinement` (`b7e4adf`) · baseline worktree `mle-star_vanilla` (`eb59252`)  
**Primary model used in eval:** `openai/gpt-5.4-mini` (seed 42)

This document is a concise **current-state report** for our skrub-integrated MLE-STAR prototype. For chronological implementation detail, see `[WORKING_PROGRESS_MLE_STAR_SKRUB.md](WORKING_PROGRESS_MLE_STAR_SKRUB.md)`. For phase-1 numbers, see `[experiments/phase1/results/phase1_summary.md](../experiments/phase1/results/phase1_summary.md)`.

---

## Executive summary

We forked Google’s MLE-STAR multi-agent pipeline and extended it in two directions:

1. **Skrub DataOps integration** — generated tabular ML code is steered toward declarative skrub pipelines (`skrub.var` → holdout bind → `.skb.apply` → `make_learner`) via ADK skills, stage prompts, and runtime compliance checks.
2. **Novel improvements** — a **data-aware refinement loop** (`skrub.TableReport` + structural ablation; TableReport used as context for ablation agent loops) and a **tuning stage** (tuning agents (plan, search, bake); in-graph `choose_*` / `choose_from` → skrub randomized search → bake).

The system runs **end-to-end** on real tabular tasks (regression and classification) with OpenAI-compatible models and a bootstrapped **vanilla MLE-STAR** baseline for controlled comparison.

**Phase-1 evaluation (12 runs, complete)** shows skrub-full as a **credible operational upgrade** (faster Python/wall time, ~65% DataOps adherence, structured pipelines) but **not a holdout score win** vs vanilla on our two tasks. Tuning and TableReport are **implemented and active** but still need hardening before they consistently add value over structural refinement.

---

## Project goals (original → current)


| Goal                                                      | Status                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Integrate skrub DataOps into all relevant MLE-STAR stages | **Done** — skill + prompts + gates; ~64–71% adherence in phase 1                |
| Add novelty to potentially improve MLE-STAR               | **Prototype done** — TableReport + terminal tuning; score gains unproven so far |
| Evaluate vs vanilla MLE-STAR on example tasks             | **Phase 1 done** — 2 tasks × 2 systems × 3 repeats; see results below           |
| Scale to broader benchmark (e.g. ~10 tasks)               | **Planned** — budget-limited; selective re-runs after open fixes                |


---

## 1. Core architecture & runtime

### Cross-provider compatibility

We have a **fully running, end-to-end prototype** with model-aware routing:

- **Gemini** → ADK `google_search`
- **OpenAI / ChatAI / GPT-5** → DuckDuckGo-backed `ddg_web_search`
- GPT-5 family temperature normalized to provider-safe values (`config.get_compatible_temperature`)

Relevant code: `shared_libraries/search_tool_util.py`, `shared_libraries/config.py`, `agent.py`.

### Native skrub integration

Skrub is wired into the pipeline through:

- **ADK `SkillToolset`** — `skrub-dataops-pipeline` skill with on-demand reference docs (holdout binding, tuning patterns, common failures, encoding, dataops pipeline, general skrub api, etc.)
- **Stage prompts** — one-liners and contracts in init, refinement, ensemble, submission, tuning, and debug agents
- **Deterministic compliance checks** in `code_util.py` / `debug_util.py` — e.g. block tool-only turns, fake tuning handoffs, ablation contract violations, tune bake integrity

Pipeline order (unchanged from MLE-STAR): **initialization → refinement → [tuning] → ensemble → submission**.

Skrub-full config flags (vs vanilla):

- `table_report_enabled` — TableReport profile into refinement ablation/planning
- `tuning_enabled` — terminal hyperparameter search stage

### Vanilla baseline

For fair comparison we bootstrapped `mle-star_vanilla` via `scripts/bootstrap_vanilla_baseline.sh` (same ChatAI/DDG/GPT-5 infra, **no** skrub skill, TableReport, or tuning). See `VANILLA_BASELINE.md` in the vanilla worktree.

---

## 2. Novelty: data-aware refinement & terminal tuning

This replaces much of the old blind trial-and-error ablation with structured, data-informed iteration.

### Semantic data auditing (`skrub.TableReport`)

At refinement start we build a compact data profile from `train.csv` and inject it into ablation and planning prompts (`{data_profile}`). Full JSON is archived as `table_report.json`.

**Phase-1 outcome:** Present on all skrub runs (~1.2k char summary → ablation agent). Helped steer housing structural ablations (refine promoted on run2–3); on Spaceship Titanic, ablation ran but **no score gain** over init — useful context, weak promotion when drift (mostly through bugs) interfered.

### Structural refinement + ablation

Refinement is steered towards **structural-only** (feature engineering / preprocessing), not backbone swaps and expensive hyperparam tuning. Ablation scripts must print `Ablation[<variant>] <metric>: …` (≥2 variants) or fail a runtime contract gate.

### Terminal tuning (`skrub.choose_`*)

Dedicated tuning subagents declare search spaces in the DataOps graph; skrub resolves choices and runs randomized search (Optuna backend), then **bakes** literals into a final script.

Hardening shipped (P1–P8):

- **Pattern 4** `choose_from` for non-sklearn estimators (e.g. CatBoost)
- Search → bake integrity gates; honest skip on search failure
- Search compute budget (reduced capacity during search, full at bake)
- Task-general metric wording (P8) — competition metric from `task_description.txt`, not hardcoded RMSE
- Value-aware `map_tuning_best_params` — fixes `data_op__N` key order ≠ plan order scramble

**Phase-1 outcome:** Tuning **mechanically works** (search + bake complete on all skrub runs) but `**tune_winner_source=structural` every time** — search never beat promoted refine on holdout. Early classification runs used wrong tune metrics (fixed in P8 prompts; archived runs pre-fix).

---

## 3. Robustness & anti-drift

We engineered guardrails so agents fail fast instead of silently degrading the pipeline under error pressure.


| Mechanism                       | Purpose                                                                                                                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ablation print contract (P7)    | Force meaningful multi-variant ablation output                                                                                                                                               |
| Tune search / bake gates (P1)   | No fake “tuned” state without parsed params or explicit skip                                                                                                                                 |
| Anti-drift debug contract       | Minimal diff; no Block 2 (full trainset retrain, test pred) in early stages; preserve variants/search nodes                                                                                  |
| Backbone prompt injection (P1c) | Debug told to keep same estimator family; Regex deterministic check to avoid promoting code unnessarily switching estimator during debug (needs to be refined)                               |
| Context bloat fixes             | Empty `code_block` guards, state truncation, JSON-only plans — avoid `ContextWindowExceededError`                                                                                            |
| FE preservation through tune    | Prompts require structural FE verbatim; tune cannot collapse to bare vectorizer+model                                                                                                        |
| Skill failure library           | `common_failure_fixes.md` (+ related refs) expanded from **observed runtime mistakes** — agents load on demand; not a fully automated self-learning loop, but a curated, growing fix catalog |


### Remaining drift gap (highest-priority open item)

**Backbone drift** still occurs in refinement/debug (e.g. HGB → LogisticRegression / RF on Titanic). Promotion gates often save holdout scores, but debug cost stays high.

Current regex misses names like `LogisticRegression`; enforcement is **prompt-only** today. A **deterministic backbone gate** is designed (~50 LOC: extended regex + set-equality check in `get_code_from_response` for debug / plan_implement / tune) — **not yet shipped**. See WORKING_PROGRESS §39.

### Validation leakage (both systems)

Parsed `Final Validation Performance` is **not trustworthy without script audit**. Documented patterns:

- Full-train fit then score holdout (vanilla housing run2) -> vanilla already has this problem
- Cross-split ensemble members + val calibration (skrub housing run1)
- Full-data `skrub.var` before split (mitigated in docs and mostly works now; runtime checker still off)

`use_data_leakage_checker=False` remains disabled until checker covers these patterns. Data-leakage checker will/should be enabled for final experiments; which will probably remove this problem (skrub-version: data-leakage agent also skrub-enabled with specific skill reference doc for leakage patterns with skrub var binding).

---

## 4. Automated evaluation pipeline

### Tooling


| Script                           | Role                                                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `test-scripts/analyze_run.py`    | Per-run report: stage scores, gains, DataOps adherence, debug counts, wall time, Python exec by stage, tune winner |
| `test-scripts/aggregate_runs.py` | Batch CSV over `experiments/phase1/`                                                                               |


Protocol: `[docs/experiment_plan_evaluation.md](experiment_plan_evaluation.md)`  
Artifacts: `[experiments/phase1/](../experiments/phase1/)` — `manifest.csv`, `results/phase1_all_runs_12.csv`, `phase1_summary.md`

### Phase-1 experiment (complete)

**Design:** 2 tasks × 2 systems × 3 repeats = **12 runs**


| Task               | Metric     | Systems               |
| ------------------ | ---------- | --------------------- |
| California housing | RMSE ↓     | vanilla vs skrub-full |
| Spaceship Titanic  | accuracy ↑ | vanilla vs skrub-full |


**Operational hypothesis (main bet):** skrub makes tabular pipelines easier to run — higher DataOps adherence, less Python/wall time, fewer debug rounds — **not** necessarily better holdout scores.

**Excluded from score claims (probable leakage):** housing skrub run1, vanilla run2. All Titanic runs kept.

### Headline results (clean runs)


| Signal                       | Housing (n=2)                    | Titanic (n=3)                |
| ---------------------------- | -------------------------------- | ---------------------------- |
| Holdout scores               | ~~tie (~~52–56k RMSE)            | vanilla **+2.5 pp** accuracy |
| Python exec                  | skrub **~2.6× faster**           | skrub **~42% faster**        |
| Wall time (full ADK session) | skrub **~2.3× shorter**          | skrub **~18% shorter**       |
| DataOps adherence            | **~0.71** vs ~0                  | **~0.64** vs ~0              |
| Fewer debug rounds           | **No** (more skrub refine debug) | **No**                       |
| Tuning beat structural       | —                                | **No** (always structural)   |


**Bottom line:** Skrub-full wins on **speed and structure**; does **not** win on **scores** (Titanic) or **debug churn** (yet). Housing is a tie on honest bands.

### Cost planning (estimate)

From ADK log LLM call counts (no token logging in artifacts yet):

- ~**€0.35–0.55** / vanilla run  
- ~**€2–3** / skrub-full run (heavy debug up to ~€5)

At **€25/month**, a full 10-task × 2-variant × 2–3 run matrix is **not feasible** without selective re-runs after backbone + leakage fixes. Instead we will start full-scale evaluation of 10 tasks conserviatively: 1 run per task x variant until 10 tasks reached, only then scale up to 2-3 runs. If openai ressources empty before 10 tasks reached, switch to ChatAI (e.g. mistral model) for both variants (maintain comparability) and clearly state in experiment analysis. 

---

## What works well today

- End-to-end tabular runs on **regression and classification** with skrub DataOps graphs
- **TableReport → ablation/planner** context on new tasks (missingness, cardinality signals)
- **Terminal tuning pipeline** (search, bake, param mapping) with integrity gates
- **Faster execution** vs vanilla on phase-1 tasks (especially housing refinement/ensemble)
- **Reproducible eval** — archived runs, manifest, automated analysis scripts
- **Vanilla baseline** worktree for apples-to-apples comparison

## Known limitations

- Holdout scores often **misleading** without leakage audit (both systems)
- **Backbone drift** under debug/refine (LLM drift under high pressure) — prompt help only; gate pending (while gate may negatively influence runtime and restrict exploring too much)
- **Tuning rarely improves** holdout vs structural refine; may disable or refocus (e.g. FE search, not only model hyperparam search as currently)
- **More debug rounds** on skrub in phase 1, not fewer — opposite of initial operational hope
- Skrub-full runs cost **~4–6×** more LLM calls than vanilla (due to debug, while debug is able to fix errors and is skrub-enabled (skill docs, web search to skrub api if not fixable with skill only)

---

## Next steps (recommended order)

1. **Ship backbone drift gate** — extended estimator detection + exec failure on family change (mostly debug; possibly also plan_implement, tune).
2. **Leakage guards** — prompt/exec rules + optional flags in `analyze_run.py`; enable `use_data_leakage_checker` when patterns A–C are covered. Will enable leakage-checker for full-scale evaluation, should reduce leakage drastically.
3. **Selective re-runs** — 1–2 tasks post P8 + backbone gate; not full 12-run grid until tune/refine stabilize.
4. **Optional:** LiteLLM token usage in `meta.json`; Tier-B external holdout under `experiments/phase1/eval/`.
5. **Scale eval** — expand task count (run count) strategically and within credit budget - switch to ChatAI if not feasible.

---

## Key references


| Document                                                                                          | Content                                   |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `[WORKING_PROGRESS_MLE_STAR_SKRUB.md](WORKING_PROGRESS_MLE_STAR_SKRUB.md)`                        | Full implementation chronology (§1–§40)   |
| `[MLE_STAR_AGENT_EXPLAINED.md](MLE_STAR_AGENT_EXPLAINED.md)`                                      | Agent pipeline architecture               |
| `[experiment_plan_evaluation.md](experiment_plan_evaluation.md)`                                  | Eval protocol and checklist               |
| `[experiments/phase1/results/phase1_summary.md](../experiments/phase1/results/phase1_summary.md)` | Phase-1 tables, timing definitions, TL;DR |
| `[todo.md](todo.md)`                                                                              | Open engineering tasks                    |


---

## TL;DR

We built a **production-shaped skrub MLE-STAR prototype**: cross-provider runtime, native DataOps skills (up to here already reported in previous status-report), data-aware refinement (TableReport-Novelty), terminal skrub tuning (Novelty), and strong runtime guardrails. Phase-1 vs vanilla shows **clear operational wins** (speed, structure, DataOps adherence) but **no score win** and **more refinement debug**, with tuning not yet beating structural solutions. Immediate focus: **backbone drift gate**, **leakage hardening (leakage checker enable - already has skrub skill for leakage checks with dataops pipelines)**, then **selective re-runs** before scaling to more tasks. Full Benchmark still open.