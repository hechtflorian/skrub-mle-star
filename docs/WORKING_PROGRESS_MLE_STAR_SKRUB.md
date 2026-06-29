# Working Progress: MLE-STAR + skrub + OpenAI/ChatAI

## Scope and current goal
- Project focus: `agents/machine-learning-engineering/`
- Main direction: run MLE-STAR on OpenAI-compatible providers (ChatAI/OpenAI), enforce `skrub` DataOps usage in generated ML code, and keep Gemini compatibility.

## Progress summary

### 1) Integrated Open-weights/ChatAI routing and web search compatibility
- Added non-Gemini search routing to DuckDuckGo-backed function tool and kept Gemini on ADK `google_search`.
- Relevant code:
  - `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/search_tool_util.py` (`ddg_web_search`, `get_search_tools`)
- Behavior:
  - Gemini models -> `google_search`
  - Non-Gemini/OpenAI-compatible models -> `ddg_web_search`

### 2) Started Skrub Injection: Prompt hardening for skrub DataOps (iterative)
- Prompts were rewritten to require DataOps pipeline structure (not only incidental `TableVectorizer` usage).
- Added compact web-search protocol for uncertain `skrub` APIs using targeted `site:skrub-data.org` queries.
- Relevant files:
  - `agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/prompt.py`
    - `MODEL_RETRIEVAL_INSTR`
    - `MODEL_EVAL_INSTR`
    - `BUG_REFINE_INSTR`
    - `CODE_INTEGRATION_INSTR`
  - `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_prompt.py`
    - `BUG_REFINE_INSTR`
  - Additional DataOps guidance across:
    - `sub_agents/refinement/prompt.py`
    - `sub_agents/ensemble/prompt.py`
    - `sub_agents/submission/prompt.py`

### 3) Integrated OPENAI API models: GPT-5 temperature compatibility fix (model-scoped)
- Added model-aware temperature normalization:
  - GPT-5 family only -> force `temperature=1.0`
  - all other models unchanged (including `openai/openai-gpt-oss-120b`, mistral, gemini)
- Relevant code:
  - `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/config.py`
    - `is_gpt5_family_model()`
    - `get_compatible_temperature()`
  - Call-site updates in:
    - `machine_learning_engineering/agent.py`
    - `sub_agents/initialization/agent.py`
    - `sub_agents/refinement/agent.py`
    - `sub_agents/ensemble/agent.py`
    - `shared_libraries/debug_util.py`
    - `shared_libraries/check_leakage_util.py`

## Important reference snippets

### A) Search routing
```23:53:agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/search_tool_util.py
def ddg_web_search(query: str, max_results: int = 5) -> str:
    ...
def get_search_tools(model_name: str) -> list:
    if is_gemini_model(model_name):
        return [google_search]
    return [ddg_search_tool]
```

### B) GPT-5 temperature guard
```45:63:agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/config.py
def is_gpt5_family_model(model_name: str) -> bool:
    ...
    return normalized.startswith("gpt-5")

def get_compatible_temperature(model_name: str, requested_temp: float) -> float:
    if is_gpt5_family_model(model_name):
        return 1.0
    return requested_temp
```

### C) Root agent temperature application
```65:74:agents/machine-learning-engineering/machine_learning_engineering/agent.py
root_model = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
...
generate_content_config=types.GenerateContentConfig(
    temperature=config.get_compatible_temperature(root_model, 0.01),
),
```

## Errors encountered and quick fixes

### 1) `.env` changes not taking effect
- Symptom: model/API values in logs did not match edited `.env`.
- Cause: shell/process env not reloaded; server already running with stale env.
- Quick fix:
  - restart `adk web` in a clean terminal
  - `set -a; source .env; set +a`
  - verify with `echo $ROOT_AGENT_MODEL` before launch

### 2) `ddg_web_search` import failure
- Symptom: `Web search failed: No module named 'ddgs'`
- Cause: runtime env missing `ddgs` package.
- Quick fix:
  - `uv sync --project "agents/machine-learning-engineering"`
  - ensure `uv run --project ... adk web` is used

### 3) GPT-5 `temperature` unsupported
- Symptom: LiteLLM error for GPT-5 with `temperature=0.01`/`0.0`.
- Cause: GPT-5 provider constraints in LiteLLM mapping.
- Fix applied:
  - model-scoped temperature normalization (GPT-5 only).

### 4) Mixed runtime path confusion
- Symptom: stack traces from a different repo/venv path than expected.
- Cause: multiple repos with same package name and mixed environment resolution.
- Quick fix:
  - pin runs with explicit project: `uv run --project "<path>" adk web`
  - verify import path before run:
    - `python -c "import machine_learning_engineering.agent as a; print(a.__file__)"`


## Known open items
- Prompt tuning is still iterative; DataOps enforcement improved but not final.
- Dedicated `skrub` retrieval skill/tool is still pending and very likely needed for stronger API reliability.

## Suggested next steps
1. Run 3-5 controlled benchmark runs (same task/model) and compare DataOps adherence in generated code.
2. Add lightweight verification checks for generated code (presence of DataOps primitives and no sklearn-only fallback).
3. Implement a focused `skrub` retrieval tool (agent skill) to reduce persisting API hallucinations.

## Progress update (latest)

### 4) Added native ADK skill wiring for `skrub-dataops-pipeline`
- Upgraded and aligned to native ADK Skills usage (`SkillToolset` + `load_skill_from_dir`) and removed reliance on fallback shim behavior.
- Added/updated helper:
  - `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/skill_tool_util.py`
- Skill package and references expanded:
  - `agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/SKILL.md`
  - `agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/*.md`

### 5) Initialization + refinement integration coverage tightened
- Initialization agents with code-generation paths now have access to skill tools at run/debug stages where needed.
- Refinement agents with planning/code-generation now have skill access (ablation, extract/plan, implement-plan flows).
- Shared debug utility supports passing tools into run-stage agents while preserving defaults.
- Relevant code:
  - `sub_agents/initialization/agent.py`
  - `sub_agents/refinement/agent.py`
  - `shared_libraries/debug_util.py`

### 6) Prompt strategy refined to reduce bloat
- Rewrote skrub-specific prompt additions to compact guidance:
  - load skill first (`list_skills` -> `load_skill`)
  - load one focused `load_skill_resource` for uncertain API details
  - fallback to `site:skrub-data.org` search only when still uncertain
- Applied only to skrub-modified prompts (did not rewrite unrelated vanilla prompts).
- Relevant files:
  - `sub_agents/initialization/prompt.py`
  - `shared_libraries/debug_prompt.py`
  - `sub_agents/refinement/prompt.py`

### 7) Added flow/control documentation for current architecture
- Created flow map with agent-by-agent instruction/wiring overview:
  - `docs/MLE_STAR_INIT_REFINEMENT_SKRUB_MAP.md`
- Added phase-1 validation notes:
  - `docs/SKRUB_SKILL_PHASE1_VALIDATION.md`

### 8) Tool-call response problem in refinement/ensemble loops (root cause + fix)
- **Observed issue:** tool-enabled agents (`ablation`, `plan_implement`, `ensemble_plan_implement`) sometimes emitted only tool calls (`list_skills`, `load_skill`) and no final code text in that turn.
- **Root cause:** response extraction keeps only text parts; tool-call-first turns can be empty text. Empty/prose output was then treated as executable artifact in some loop stages, causing false progress and downstream score errors (e.g., `KeyError: 'score'`).
- **Specific fixes applied (surgical):**
  - `shared_libraries/code_util.py`:
    - added execution gates in `get_run_code_condition(...)` for `ablation*`, `plan_implement*`, and `ensemble_plan_implement*`:
      - run only if non-empty output
      - run only if syntactically valid Python (`compile(...)`)
  - `sub_agents/refinement/agent.py`:
    - `check_plan_implement_finish(...)` now finishes only on successful scored result (`returncode == 0` and `"score"` present)
    - refinement score aggregation skips invalid entries missing scores instead of indexing blindly
    - empty plan-refine responses are ignored (not appended)
  - `sub_agents/ensemble/agent.py`:
    - `check_ensemble_plan_implement_finish(...)` now requires successful scored execution before finishing
- **Prompt/skill alignment changes:**
  - strengthened tool-usage contract: tool calls are preparation only; final output must be executable code
  - refinement prompts were tuned to require focused resource loading (not always loading all references)
  - ensemble implementation prompt now requires a concrete `load_skill_resource` call sequence before coding
- **Current state:** refinement and ensemble loops are more robust against tool-only turns; invalid tool-first outputs no longer advance code execution stages as if they were valid artifacts.

### TL;DR
- The failure was not the skill itself, but treating tool-call/empty responses as valid code artifacts.
- We fixed this at the runtime boundary with minimal gates and stricter finish criteria.
- Result: agents now wait for real executable code + scored execution before progressing, which removed the recurring refinement/ensemble loop breaks.


# Status Report & Experiment Plan (01.06.2026)
## Development

### Current prototype status
- Current prototype supports OpenAI-compatible runtime + Gemini compatibility, model-aware GPT-5 temperature handling, and model-aware search routing.
- `skrub` is now integrated via native ADK skill loading and targeted prompt directives in initialization and refinement stages.
- The prototype is functionally usable for end-to-end runs, but results quality is still mixed and requires structured evaluation at scale.

### Future development plan (open items and improvements)
- Expand from initialization/refinement-focused integration to stronger consistency checks across ensemble/submission phases.
- Add lightweight automatic run-time checks that score DataOps compliance and flag sklearn-only fallback patterns.
- Improve refinement loop quality by encouraging DataOps-native tuning paths (e.g., `choose_from`, DataOps search methods) and reducing non-DataOps regressions.
- Standardize and version the skill/reference package so updates are tracked with benchmark impact.

### Two biggest open challenges
1. **Reliability of DataOps adherence in generated code**  
   Even with skill + prompt constraints, agents can still drift into partial or superficial skrub usage under error pressure.
2. **Measuring true quality gains versus baseline MLE-STAR**  
   Need controlled, repeated experiments to separate gains from randomness (model variance, task variance, search variance).

## Experiment plan

### Experiments to run (at least 10 tasks)
- Evaluate on at least 10 Kaggle tabular tasks (regression and classification mix).
- For each task, run both variants:
  - **Baseline**: vanilla MLE-STAR (no skrub skill enforcement).
  - **Skrub variant**: current branch with skrub skill + prompt/refinement improvements.
- Per task, run 3 seeds (or 3 repeated runs where seeds are fixed but runtime variability exists).
- Submit final `submission.csv` from each run to Kaggle for leaderboard metric comparison.
- Store per-run artifacts:
  - generated code files
  - `final_state.json`
  - logs/tool-call traces
  - submission metric and runtime metadata

### How to measure benefit (concrete metrics)
- **Primary outcome metric (task quality):**
  - Kaggle public score delta (Skrub variant - Baseline) per task.
  - Win-rate: % tasks where Skrub variant beats baseline.
- **Secondary quality metrics:**
  - DataOps adherence score (regex + structure checks), e.g. presence/use of:
    - `skrub.var` / `skrub.X` / `skrub.y`
    - `.skb.apply(...)`
    - `choose_*` / `choose_from(...)`
    - absence of full sklearn-only orchestration fallback
  - Debug efficiency:
    - number of rollback/debug rounds
    - % runs reaching successful execution without manual intervention
  - Cost/performance:
    - tokens consumed
    - wall-clock runtime
    - cost per successful valid run

### Expected improvements (concrete range)
- **DataOps adherence:** expected strong increase, roughly +25% to +60% in adherence score versus baseline.
- **Leaderboard performance:** expected modest-to-moderate average gain, roughly +0.5% to +3% relative improvement on normalized task metric; some tasks may be neutral or slightly worse.
- **Robustness:** expected reduction in DataOps-related API hallucination/debug churn by ~10% to ~30% after skill stabilization.

### Mitigation plan for problems and resource shortages
- **Tokens/cost pressure:** cap retries, shorten context, enforce focused skill-resource loading (one reference at a time), and downshift model for ablation loops.
- **Runtime/timeouts:** apply early-stop policies, set per-stage max runtime budgets, and prioritize top-N candidate runs.
- **Unexpected regressions:** keep baseline fallback branch; if Skrub variant underperforms repeatedly on a task family, use task-specific gating rules.
- **Sparse or noisy results:** increase repeats for unstable tasks and report confidence intervals, not only single-run scores.
- **Operational failures:** maintain run-resume checkpoints and standardized run manifests to recover without losing full experiments.

---

## Progress update (2026-06-06): separate tuning stage, TableReport ablation context, run analysis

### 9) Refinement reverted to structural-only; tuning extracted to own module
- Removed embedded tuning from `sub_agents/refinement/agent.py` (structural ablation → plan → implement only).
- Added `sub_agents/tuning/agent.py` + `prompt.py` with flow:
  - `tune_plan` → `tune_implement` → `tune_bake` → `promote_tuning_winner`
- Wired into pipeline in `machine_learning_engineering/agent.py` when `config.CONFIG.tuning_enabled`.
- Config renamed: `tuning_enabled`, `tuning_n_iter`, `tuning_n_jobs` (replacing `refinement_tune_*`).
- Handoff unchanged for ensemble: reads `train_code_{outer_loop_round}_{task_id}`; on tuning win, keys overwritten + `train{N}_tuned.py` written.

### 10) TableReport data profile for ablation agent
- Added `shared_libraries/table_report_util.py`:
  - `load_table_report_dict(train.csv)` via `skrub.TableReport`
  - `format_ablation_profile(...)` — compact column stats, missingness, top associations
- `init_outer_loop_states` in refinement builds `ablation_table_report_profile_{task_id}` and writes `workspace/<task>/table_report.json`.
- Injected into ablation prompts as `{data_profile}` with hints for encoding/routing/redundancy ablations (`refinement/prompt.py`).

### 11) Tuning implement loop fixes (post `adk_run_20260606_211508`)
- Root cause of tune stuck loop: `json.dumps(best_params)` failed on numpy scalars → `returncode=1` → finish gate never passed.
- Fixes in `shared_libraries/code_util.py`:
  - `normalize_tuning_best_params()`, `extract_tuning_best_params()`
  - Restored tune gates: `make_randomized_search`, `search.fit`, `TUNING_BEST_PARAMS` line
  - Restored `tune_bake` placeholder check (`choose_*` / search calls)
- `tuning/prompt.py`: require `search.fit`, `json.dumps(..., default=str)`, no prose-only replies.
- `choices_hparam_pattern.md`: terminal search pattern + JSON-safe params.
- `tuning/agent.py`: `check_tune_implement_finish` aligned with refinement (`returncode==0` + score).

### 12) Successful end-to-end validation run (`adk_run_20260606_223108`)
- Full pipeline completed: init → refinement → tuning → ensemble → submission.
- Artifacts under `workspace/california-housing-prices/`:
  - Structural winner: `1/train1.py` (= `train0_improve0.py`, geo features + encoders + LightGBM, holdout RMSE ~2664)
  - Tuning ran but lost: search ~57712, bake ~11021 → `tune_winner_source_1: structural`
  - Final submission holdout ~2414 (ensemble stack)
- Compared to older `gpt-5.4-mini` submissions (~54k train RMSE, CatBoost-heavy): latest run is genuinely better pipeline/code, not a scoring bug.

### 13) Architecture / memory analysis (documented, not implemented)
- No shared conversational memory across subagents (`include_contents="none"`); state keys are the handoff.
- Debug path keeps **latest** `bug_summary_*` only — no failure history → agents can repeat failed fixes.
- Recommended direction: **state-based experiment ledger** + stdout truncation, not full chat memory.
- Captured in **`docs/MLE_STAR_FUTURE_IMPROVEMENTS.md`** (prioritized backlog for coding agents).

### 14) Known issues to address next (see future improvements doc)
- `final_state.json` bloated (~6 MB) by tune search stdout (80k× LightGBM warning lines); truncate before state storage.
- Ablation often uses a **different pipeline** than `train{N}.py` → misleading summaries for planners.
- Scores across stages not comparable without unified validation harness.
- Tuning stage: architecturally good, empirically neutral/negative on first California Housing win — needs hardening (`verbose=-1`, subsample search, deterministic bake).

### TL;DR (June 2026)
- Tuning is a separate stage with promotion gate; implement loop fixed and validated end-to-end.
- TableReport gives ablation structured dataset context; impact on simple tabular tasks is modest — consider feeding planners too.
- Next work is correctness/comparability (same-pipeline ablation, unified holdout, stdout caps) before more prompt expansion.

---

## Progress update (2026-06-07 / 2026-06-08): holdout correctness, tuning hardening, leakage checker

This section brings the doc in line with the current prototype after refinement/tuning iteration, skill-reference work, and California Housing validation runs (`r4`).

### Prototype pipeline (current)

```
init → refinement (structural) → [tuning] → ensemble → submission
         │ ablation + plan/implement          tune_plan → tune_implement → tune_bake → promote
         └ TableReport profile              (optional; config.tuning_enabled)
```

**Novelty vs vanilla MLE-STAR:** skrub DataOps is enforced via ADK skills; refinement is **structural-only** (no in-loop hyperparameter search); **terminal tuning** is a dedicated stage with in-graph `choose_*`, holdout randomized search, bake-to-fixed-params, and a promotion gate. Ensemble/submission consume fixed-parameter pipelines only.

### 15) Skill reference package expanded (refinement + tuning guidance)

Beyond the original `dataops_api_quickmap.md`, the skill now ships focused references agents load on demand:

| Reference | Purpose |
|-----------|---------|
| `references/choices_hparam_pattern.md` | In-graph `choose_*`, holdout search, bake handoff |
| `references/feature_engineering_skrub.md` | Derived features, ablation contract, post-FE routing |
| `references/encoding_skrub.md` | Encoders / `TableVectorizer` |
| `references/selectors_routing_skrub.md` | Column routing, split→concat |
| `references/common_failure_fixes.md` | Runtime failures + fake tuning |
| `references/dataops_tuning_optuna.md` | Optuna backend (advanced; not default path) |
| `references/holdout_data_leakage.md` | **Leakage checker only** — audit bind/fit patterns |

**`SKILL.md`** — default template is **holdout-only** for early stages (`train_part` bind → metric print). Full-train + test export moved to a separate **submission-stage** note (see §22 below; supersedes the earlier two-block default template).

**Prompt alignment (minimal one-liners, no bloat):**
- `sub_agents/initialization/prompt.py` — load quickmap + holdout-only early stages
- `sub_agents/refinement/prompt.py` — ablation/implement fit on `train_part` only; no `test_df`
- `sub_agents/tuning/prompt.py` — bake/search holdout metric only (no test/submission in tune scripts)
- `shared_libraries/debug_prompt.py` — preserve holdout binding when fixing skrub code

### 16) Holdout data leakage — problem, impact, fix

**Problem discovered:** Generated scripts often bound `skrub.var("data", train_df)`, split into `train_part` / `valid_part`, then called `make_learner(fitted=True)` and `predict({"data": valid_part})`. Because fit uses **all** bound rows, validation rows leak into training → **optimistic** `Final Validation Performance` and wrong promotion decisions.

**Evidence (California Housing `r4`):**
- Structural / init holdout RMSE ≈ **24 293** (leaky bind on full `train_df`)
- Ablation holdout RMSE ≈ **57 088** (often binds `train_part` correctly in ablation variants)
- Same split (`test_size=0.2`, `random_state=42`) — large gap indicates metric incomparability, not real model gain

**Fix (docs-first, no runtime gate yet):**
- `references/dataops_api_quickmap.md` — canonical holdout section + anti-pattern table
- `references/choices_hparam_pattern.md` — search/bake must use `train_part` for metric path
- `references/common_failure_fixes.md` — item **#15** (holdout leakage via full-data `skrub.var`)
- `references/feature_engineering_skrub.md`, `selectors_routing_skrub.md` — examples use `train_part` in Block 1

**Correct pattern (early stages — holdout metric only):**

```python
data_train = skrub.var("data", train_part)
# ... build graph on data_train ...
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
print(f"Final Validation Performance: {rmse}")
# stop here in init / ablation / refinement / tuning
```

**Submission stage only** (full train + test — not copied into early scripts; see §22):

```python
data_full = skrub.var("data", train_df)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
```

**Still open:** `use_data_leakage_checker` defaults to `False` in `config.py`; enable when ready to auto-audit/refine. Tuning search scripts (`train_tune_search.py`) are excluded from promotion — **`train_tune_baked.py`** is the promoted artifact.

### 17) Tuning runtime fixes (`code_util.py` + `tuning/agent.py`)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Numpy in `TUNING_BEST_PARAMS` | `json.dumps` crash → tune loop stuck | `normalize_tuning_best_params`, prompt requires `default=str` |
| Generic `data_op__N` keys in state | `tune_best_params` unreadable vs plan | `map_tuning_best_params(raw, tune_plan)` maps by `tunable_params` order |
| Strict tune_implement finish | Loop never exits on partial success | Finish on `returncode==0` (aligned with original MLE-STAR); search stdout may still show `data_op__*` |
| Bake still has `choose_*` | Downstream runs search again | `code_contains_tuning_placeholders()` gate on bake finish |
| Tune search vs bake eval mismatch | Incomparable scores | Skill + prompts: bake uses same holdout protocol as structural (Block 1) |

**Mapping snippet (`code_util.py`):**

```python
def map_tuning_best_params(raw: dict, tune_plan: dict) -> dict:
    tunable = tune_plan.get("tunable_params") or []
    names = [p.get("name") for p in tunable if p.get("name")]
    ordered_values = [v for _, v in sorted(raw.items(), key=lambda x: _data_op_sort_key(x[0]))]
    return {names[i]: ordered_values[i] for i in range(len(names))}
```

**Verify on next run:** `final_state.json` → `tune_best_params_{task}` should have human names (`learning_rate`, `max_depth`, …), not `data_op__0`. Compare literals in `train_tune_baked.py`.

**Tuning stage skips leakage checker** on purpose: `tune_implement_skip_data_leakage_check_*` and `tune_bake_skip_data_leakage_check_*` set in `tuning/agent.py` (search script is diagnostic; bake promoted after skill-aligned holdout).

### 18) Data leakage checker agent wired to skill

Optional sequential sub-agent (refinement ablation path + debug path when `config.CONFIG.use_data_leakage_checker=True`):

```
check_leakage_loop → refine_leakage (patch leaky block in place)
```

**Files:**
- `shared_libraries/data_leakage_prompt.py` — one-liner to load `references/holdout_data_leakage.md`
- `shared_libraries/check_leakage_util.py` — `tools=[skill_tool_util.get_skill_toolset()]` on check + refine agents
- `skills/.../references/holdout_data_leakage.md` — checker-focused rules, leakage signal table, fix contract

Checker agents load **only** the leakage reference (not full quickmap) to limit context bloat.

### 19) Run analysis tooling

Added `test-scripts/analyze_run.py` (mirror under `eval-logs/`) — parses `final_state.json`, workspace scripts, and ADK logs:
- Stage scores (init / refinement / tuning / ensemble / submission)
- DataOps anchor regex checks per script
- Tuning novelty flags (`choose_*`, `TUNING_BEST_PARAMS`, `tune_winner_source`)
- Skill resource load traces

Example:

```bash
python test-scripts/analyze_run.py \
  --state submissions/.../r4/final_state.json \
  --workspace submissions/.../r4
```

### 20) California Housing results snapshot (`r4`, `gpt-5.4-mini`, tuning enabled)

Artifacts: `submissions/california-housing-prices/gpt-5.4-mini/tuning-agents-added/r4/`

| Stage | Artifact | Holdout RMSE (approx.) | Notes |
|-------|----------|------------------------|-------|
| Init winner | HGB + ratio features | 24 293 | Leaky full-`train_df` bind (pre-holdout-doc fix) |
| Ablation | `ablation_0.py` | 57 088 baseline | Honest `train_part` bind in variants |
| Refinement | `train0_improve1.py` | 24 293 | Structural winner before tune |
| Tune search | `train_tune_search.py` | (not promoted) | Holdout search + `TUNING_BEST_PARAMS` |
| Tune bake | `train_tune_baked.py` | **23 043** | Promoted → `train1_tuned.py` |
| Promotion | `tune_winner_source_1` | **`tuned`** | Bake beat structural on parsed score |

**Interpretation:** Tuning stage **can** win when implement+bake complete; however pre-fix holdout scores across stages were **not apples-to-apples** (leakage vs honest bind). After skill/prompt holdout update, expect higher structural RMSE and comparable ablation/refinement/tune metrics.

**`tune_plan_1` example (model focus, 3 `choose_*` on HGB):** `learning_rate`, `max_depth`, `min_samples_leaf` — frozen preprocessing from structural solution.

### 21) Refinement novelty recap (what we built)

1. **Structural refinement loop** — ablation (TableReport `{data_profile}`) → init/refine plan → plan_implement; no `choose_*` in this stage.
2. **Ablation contract** — same backbone model + same split unless hypothesis says otherwise; skill refs for encoding / FE / routing.
3. **Terminal tuning module** (`sub_agents/tuning/`) — `tune_plan` JSON schema (`focus_block`, `tunable_params`, `frozen`, `n_iter`) → bounded holdout search → bake literals → `promote_tuning_winner`.
4. **Promotion gate** — overwrites `train_code_{round}_{task}` only if baked code runs and beats structural score (`lower=True` for RMSE).
5. **Skill-first codegen** — agents call `list_skills` → `load_skill` → targeted `load_skill_resource`; tool-only turns gated in `code_util.get_run_code_condition`.

### Problems encountered → fixes (quick index)

| # | Problem | Fix location |
|---|---------|--------------|
| 1 | Tool-only agent turns treated as code | `code_util.py` compile/empty gates; finish criteria in refinement/ensemble agents |
| 2 | Tune loop stuck on numpy JSON | `normalize_tuning_best_params`, prompt `default=str` |
| 3 | `data_op__N` in `tune_best_params` | `map_tuning_best_params` + `choices_hparam_pattern.md` bake mapping note |
| 4 | Optimistic validation RMSE (skrub bind) | Skill docs holdout Block 1 + prompt one-liners |
| 5 | Ablation RMSE ≠ solution RMSE | Ablation contract + FE reference; holdout bind alignment |
| 6 | `final_state.json` bloated by tune stdout | Still open — truncate before persist |
| 7 | Block 2 copied into every stage (runtime) | §22 stage-aware skill docs + prompt alignment |
| 8 | Ensemble 5×3× CatBoost grid (runtime) | §23 ensemble prompt guardrails |

### Files touched (recent holdout + tuning + leakage work)

```
agents/.../skills/skrub-dataops-pipeline/
  SKILL.md
  references/dataops_api_quickmap.md
  references/choices_hparam_pattern.md
  references/common_failure_fixes.md
  references/feature_engineering_skrub.md
  references/selectors_routing_skrub.md
  references/dataops_tuning_optuna.md
  references/holdout_data_leakage.md          # new — leakage checker

agents/.../shared_libraries/
  code_util.py                                 # map_tuning_best_params, tune gates
  data_leakage_prompt.py
  check_leakage_util.py                        # skill tools on leakage agents
  debug_prompt.py

agents/.../sub_agents/
  initialization/prompt.py
  refinement/prompt.py
  tuning/prompt.py
  tuning/agent.py
  ensemble/prompt.py                      # §23 runtime guardrails

test-scripts/analyze_run.py
docs/WORKING_PROGRESS_MLE_STAR_SKRUB.md        # this file
```

### Known open items (updated)

- Re-run after **§22 stage-aware skill update**; early scripts should match `ablation_0.py` pattern (metric only, no `test_df`).
- Confirm `map_tuning_best_params` in `final_state.json` on fresh run.
- Enable `use_data_leakage_checker=True` once holdout docs stabilize.
- Truncate tune search stdout before state write; feed TableReport to planners (not only ablation).
- Unified validation harness / same-pipeline ablation still desirable for planner trust.
- Monitor ensemble runtime after §23 prompt guardrails (prior run: 5-fold × 3-seed CatBoost grid).

### TL;DR (current prototype state)

- **MLE-STAR + skrub:** OpenAI/ChatAI-compatible runtime, native ADK `skrub-dataops-pipeline` skill, structural refinement with TableReport-guided ablation, and a **separate terminal tuning stage** (`choose_*` → search → bake → promote).
- **Holdout correctness:** Block 1 (`train_part` bind) validated in post-fix rerun; full-train + test moved to **submission only** in skill docs + prompts (§22).
- **Runtime:** ensemble prompt guardrails added (§23) after 5×3× CatBoost grid blew up wall-clock; early-stage Block 2 copy removed from templates.
- **Tuning hardening:** JSON-safe best params, `data_op__N` → plan name mapping, bake placeholder gate, promotion from **`train_tune_baked.py`** only.
- **Next:** full pipeline rerun with holdout-only early scripts; optional leakage checker; stdout/state slimming.

---

## Progress update (2026-06-10): holdout rerun validation, stage-aware templates, ensemble runtime guardrails

Follow-up after the §16 holdout skill-doc pass. User reran with leakage checker still disabled (`use_data_leakage_checker=False`); run stopped early at ensemble due to long runtime (`adk_run_20260610_113310`, workspace `california-housing-prices/`).

### 22) Stage-aware holdout templates (early = metric only; submission = full train + test)

**Problem:** The initial holdout fix put Block 1 + Block 2 in `SKILL.md` and `dataops_api_quickmap.md` default examples. Agents copied full-train refit + `test_df` + `submission.csv` into **every** stage (init, refinement, tune bake/search), doubling CatBoost training per script execution. Vanilla MLE-STAR intent: holdout scoring in dev loops; full `train_df` + test export at **submission** only.

**Fix applied:**

| Area | Change |
|------|--------|
| `SKILL.md` | Default template = holdout metric only; Block 2 relegated to “Submission stage only” pointer |
| `dataops_api_quickmap.md` | Split “Default script (early stages)” vs “Submission stage only”; Option C labeled submission-only |
| `choices_hparam_pattern.md` | Tune search/bake = holdout + params only; no test/submission in examples |
| `holdout_data_leakage.md` | Block 2 = submission-stage only; early scripts without test are not flagged |
| `common_failure_fixes.md` | #15 fix: remove test/full-train from early scripts |

**Prompt alignment (same pass):**

- `initialization/prompt.py` — holdout-only early stages
- `refinement/prompt.py` — no `test_df` / full-train refit in implement
- `tuning/prompt.py` — removed test/submission from `tune_implement` and `tune_bake`
- `ensemble/prompt.py` — holdout metric only unless plan explicitly needs test export

**Expected script shape after this update:**

```python
# init / ablation / refinement / tune — stop here
data_train = skrub.var("data", train_part)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
print(f"Final Validation Performance: {rmse}")
```

### 23) Ensemble runtime guardrails (`ensemble/prompt.py`)

**Problem (rerun):** `ensemble0.py` implemented 5-fold KFold × 3 random seeds × (holdout fit + full-train test predict) ≈ **30× CatBoost@5000 iterations** — run appeared stuck at `ensemble_plan_implement_initial_agent`.

**Fix:** One concise line per ensemble prompt stage (plan / implement / refine):
- Prefer prediction-level merge (average/stack holdout preds); avoid multi-fold, multi-seed, or repeated full retrains unless quality gain clearly justifies cost.
- Quality still matters slightly more than runtime — not a hard ban on CV, just a strong default toward lightweight ensembling.

### 24) Post-fix rerun analysis (`adk_run_20260610_113310`)

Workspace: `agents/.../workspace/california-housing-prices/`

| Check | Result |
|-------|--------|
| **Holdout Block 1** | **Working** — `skrub.var("data", train_part)` before metric in init, refinement, tune search/bake |
| **Ablation** | **Good reference** — `ablation_0.py` holdout-only, no Block 2 (model to copy going forward) |
| **Tuning search** | **Correct** — `search.fit({"data": train_part})`, `best_learner_.predict({"data": valid_part})`, `TUNING_BEST_PARAMS` |
| **Tuning bake** | Block 1 correct; baked literals applied; promoted to `train1.py` |
| **Block 2 in early stages** | **Still present in this run** (pre-§22 templates) — all `1/*.py` had full-train + `submission.csv` |
| **Ensemble** | Holdout bind OK per fold, but **5×3 retrain grid + Block 2 in loop** → runtime explosion |
| **Score comparability** | Ensemble used KFold OOF metric vs upstream `train_test_split(0.2)` — still not apples-to-apples |
| **Truncated artifacts** | `train0_improve0.py` / `train0_improve1.py` only 27 lines (header only); `train1.py` complete |

**Refinement / tuning working as intended?** Yes structurally (ablation → plan → implement → tune plan → search → bake → promote). Operationally, pre-§22 Block 2 copy and ensemble grid cost dominated runtime; §22–§23 address both.

### Files touched (2026-06-10 pass)

```
agents/.../skills/skrub-dataops-pipeline/
  SKILL.md
  references/dataops_api_quickmap.md
  references/choices_hparam_pattern.md
  references/holdout_data_leakage.md
  references/common_failure_fixes.md

agents/.../sub_agents/
  initialization/prompt.py
  refinement/prompt.py
  tuning/prompt.py
  ensemble/prompt.py

docs/WORKING_PROGRESS_MLE_STAR_SKRUB.md
```

### TL;DR (2026-06-10)

- Holdout **leakage fix confirmed** in rerun (Block 1 on `train_part`); ablation already showed the right early-stage shape.
- **New issue found:** two-block template taught agents to full-train + predict test every stage → fixed by stage-aware skill docs + prompt lines (§22).
- **Ensemble runtime:** lightweight-merge defaults added to ensemble prompts (§23) after 5×3 CatBoost grid stalled the run.
- **Next rerun** should show holdout-only scripts through tuning and faster ensemble; submission agent still owns full train + `./final/submission.csv`.

---

## Progress update (2026-06-11) — P0/P2/P3 + rerun `adk_run_20260611_165054`

### 25) Runtime guardrails shipped (P0, P2, P3, tune fail-fast)

| Change | Purpose | Files |
|--------|---------|-------|
| **P0** plan_implement no-op gates | Stop tool-only / unchanged baseline re-eval passing as refinement | `debug_util.py`, `refinement/agent.py`, `refinement/prompt.py` |
| **P2** inline `choose_*` on apply + search required in exec gate | Prevent NumericChoice in estimator kwargs; block debug scripts without search | `choices_hparam_pattern.md`, `common_failure_fixes.md` #16, `tuning/prompt.py`, `code_util.py`, `debug_util.py` |
| **P3** ablation `.apply_func` / DataOp | Reduce deferred-helper pandas mistakes | `common_failure_fixes.md` #17, `refinement/prompt.py` |
| **Tune fail-fast** | One implement attempt per rollback → debug with stderr | `tuning/agent.py` `check_tune_implement_finish` |

### 26) Rerun results (`adk_run_20260611_165054`)

Workspace: `agents/.../workspace/california-housing-prices/`

| Stage | Holdout RMSE | Verdict |
|-------|-------------|---------|
| Init | 54645.59 | OK |
| Refinement initial implement | **51830.74** | **P0 success** — ratio FE, −2815 RMSE |
| Refinement inner implement | 71685.11 (Ridge) | Debug fixed DataOp bug but **backbone drift**; not promoted |
| Tuning search | n/a | **Failed** — NumericChoice persists |
| Tuning bake | 51830.74 | **Structural re-copy**; `tune_best_params` absent; `tune_winner_source: structural` |
| Submission | 51830.74 | OK |

**What worked**

- P0: initial `plan_implement_initial` produced real code (no silent `list_skills`-only pass).
- P3: ablation completed with holdout `@skrub.deferred` ratios; good planner signal.
- Tune fail-fast: debug engaged quickly; `get_run_code_condition` rejected no-search RF scripts.
- Promotion: best-of-inner-loop kept CatBoost 51830 over Ridge 71685.

**What still needs work**

1. **Refinement inner `plan_implement`:** 10 vanilla retries on `"households" in X_train.columns` (DataOp membership) before debug — propose **fail-fast** (P1b) + skill **#18**.
2. **Tuning:** agent still binds `choose_*` to variables → NumericChoice; debug removes search anyway → **P1 bake gate** + plan-default fallback when search exhausts.
3. **Debug drift:** Ridge / RF swaps despite backbone + preserve-search prompts — tighten with deterministic gates.

See `docs/todo.md` for prioritized backlog (P1, P1b, P1c, P1d).

### Files touched (2026-06-11 pass)

```
agents/.../shared_libraries/
  debug_util.py          # P0 plan_implement gates; P2 tune debug line
  code_util.py           # P2 tune search exec gate

agents/.../sub_agents/
  refinement/agent.py    # P0 finish gate
  refinement/prompt.py     # P0, P3 ablation line
  tuning/agent.py        # tune fail-fast
  tuning/prompt.py       # P2 inline choose line

agents/.../skills/.../references/
  choices_hparam_pattern.md
  common_failure_fixes.md  # #16, #17

docs/todo.md
docs/WORKING_PROGRESS_MLE_STAR_SKRUB.md
```

### TL;DR (2026-06-11)

- **P0 worked:** refinement initial implement now changes code and score (51830 vs 54645 init).
- **P2/P3 docs + gates help** but **NumericChoice tuning bug persists** in LLM output; bake still runs without search handoff (**P1 next**).
- **Keep tune fail-fast**; add bake gate + plan-default fallback — do **not** revert to 10 blind tune retries.
- **Refinement inner implement** should adopt fail-fast too (10 blind retries on same DataOp error before debug).

---

## Progress update (2026-06-12 → 2026-06-14): P1–P7 hardening, phase-1 eval, vanilla baseline, first housing comparison

Follow-up after the June 11 run. Shipped tuning integrity + anti-drift + context fixes on `improve-refinement` (`b7e4adf`), bootstrapped a **vanilla MLE-STAR** worktree for controlled comparison, ran phase-1 California Housing (4/8 runs), and audited archived artifacts for metric validity.

### 27) P1–P7 + context fixes (skrub-full prototype, `improve-refinement`)

| Pass | What | Key outcome |
|------|------|-------------|
| **P1** | Tune search → bake integrity | Pattern 4 `choose_from` for non-sklearn estimators (CatBoost); `tune_param_source` (`search` / `plan_defaults` / `skipped`); bake gate; identity guard in `map_tuning_best_params` |
| **P1b** | DataOp eager ops | Skill #18 + refinement implement pointer |
| **P1c** | Debug backbone drift | Stronger `debug_prompt.py`; deterministic backbone injection in `debug_util` |
| **P5** | Search compute budget | Reduced capacity during search; full capacity restored at bake |
| **P7** | Ablation print contract | ≥2 `Ablation[...]` lines or synthetic exec failure |
| **Anti-drift** | Minimal-diff debug | No full regeneration, no Block 2 in early stages, no variant/search removal |
| **Context** | `ContextWindowExceededError` | Empty `code_block` guards; `truncate_for_state`; JSON-only init_plan; `verbose=0` in SKILL.md |
| **FE-drop fix** | Tuning preserves structural pipeline | Refinement/tuning prompts: reproduce FE verbatim; tune cannot simplify to bare `TableVectorizer+model` |
| **Tooling** | `analyze_run.py`, `aggregate_runs.py` | Stage scores, gains, sentinel cleaning, `to_row()` CSV export; batch walk of `experiments/phase1/` |

See `docs/todo.md`, `docs/lessons.md`, and commit `b7e4adf` for file-level detail.

### 28) Vanilla MLE-STAR baseline (git worktree + bootstrap script)

**Goal:** apples-to-apples baseline without skrub skill, TableReport, or tuning stage.

| Item | Detail |
|------|--------|
| Script | `scripts/bootstrap_vanilla_baseline.sh` |
| Snapshot | `ffa365c` — ChatAI routing, DDG search, GPT-5 temperature, safe response parsing |
| Branch | `vanilla-baseline` @ worktree `../mle-star_vanilla` (commit `eb59252`) |
| Pure sklearn prompts | Optional `--revert-prompts` → restores `6c96e03` wording + drops skrub/optuna from `pyproject.toml` |
| Marker | `mle-star_vanilla/VANILLA_BASELINE.md` records bootstrap flags |

**Operational note:** first bootstrap ran **without** `--revert-prompts`, so `debug_prompt.py` still contained ffa365c skrub/DataOps preservation lines. When init code failed on missing packages (e.g. `lightgbm`), the debug agent rewrote toward skrub even though init/refinement prompts were sklearn-only. Fix for future vanilla runs: revert `debug_prompt.py` to `6c96e03` or bootstrap with `--revert-prompts`.

### 29) Phase-1 experiment plan & archive layout

Document: `docs/experiment_plan_evaluation.md`

**Matrix:** 2 tasks × 2 systems × 2 repeats = 8 runs  
**Task 1 (in progress):** `california-housing-prices` — vanilla + skrub-full, run1/run2  
**Task 2 (pending):** `spaceship-titanic`

Archive path pattern:

```
experiments/phase1/california-housing-prices/
  vanilla/gpt-5.4-mini/run1|run2/
  skrub-full/gpt-5.4-mini/run1|run2/
```

Each run: `meta.json`, `analysis.json`, `final_state.json`, `adk_run_*.log`, workspace snapshot (`1/`, `ensemble/`).

**Parallel runs:** safe only in **separate checkouts/worktrees** or different `task_name`s — same repo shares `workspace/<task>/` and will clobber artifacts.

### 30) Phase-1 California Housing — completed runs (holdout RMSE, lower better)

All runs: `openai/gpt-5.4-mini`, `seed=42`, holdout `train_test_split(0.2, random_state=42)`.

| System | Run | Init | Refine promoted | Ensemble / final | Honest? | Notes |
|--------|-----|------|-----------------|------------------|---------|-------|
| **vanilla** | run1 | 54 744 | 54 744 (no promote) | **50 095** | Mostly | Refine improve ~51 588 not promoted; ensemble blend tuning on val (mild optimism); `had_sentinel_failure` |
| **vanilla** | run2 | 54 252 | **10 550** | **10 385** | **No** | **Full-train leak:** `fit(X,y)` then `predict(X_val)` printed as `Final Validation Performance` — reproduced exactly (54 276 → 10 550) |
| **skrub-full** | run1 | 56 663 | 56 663 | **42 010** | **No** | CatBoost missing → **HistGradientBoosting** fallback; ensemble cross-split leak (~82% val rows in member B train) + val rank/linear calibration |
| **skrub-full** | run2 | 55 450 | **52 788** | **52 740** | **Yes (best so far)** | Real refinement gain; tune search ran but structural won; ensemble modest; DataOps adherence 0.70 |

**Interpretation for comparison (use honest band ~50–56k RMSE, not inflated finals):**

- **skrub-full run2** is the first archived run with credible stage progression (refine −2.7k, ensemble stable).
- **skrub-full run1** final 42k and **vanilla run2** final 10k are **internal metric artifacts**, not trustworthy generalization estimates.
- **Vanilla run1** final ~50k is the most credible vanilla signal so far; run2 refine/final scores should be discarded.

**CatBoost vs actual model:**

- Vanilla run2: real CatBoost (pip-installed at runtime).
- Skrub-full run1: intended CatBoost, actual **HGB** via `except ModuleNotFoundError` alias; skrub `TableVectorizer` pipeline otherwise correct.
- Skrub-full run2: CatBoost available in venv; skrub DataOps + refinement FE improvements.

### 31) Leakage patterns discovered in phase-1 archives (both systems)

Common failure: agents conflate **“retrain on full train for submission”** with **“report holdout score.”**

| Pattern | Where seen | Mechanism | Symptom |
|---------|------------|-----------|---------|
| **A. Full-train → eval holdout** | vanilla run2 `train0_improve0.py`, `train1.py`, ensemble | Grid on `X_tr`, then `final_model.fit(X,y)`, then `predict(X_val)` | RMSE ~10k (impossible for this task) |
| **B. Cross-split ensemble member** | skrub-full run1 `final_solution.py` | Train member B on split seed 7, eval on `valid_part` from seed 42 | ~82.5% val rows already in B's train |
| **C. Validation-set calibration** | skrub-full run1 ensemble | Rank → val target quantiles + `polyfit` on val labels; pick blend on val | Extra optimistic RMSE (~42k) |
| **D. skrub bind on full `train_df`** | older runs (doc §16) | `skrub.var("data", train_df)` before split | Optimistic structural scores — mitigated in skill docs, still needs runtime checker |

**Reproduction anchors:**

- Vanilla run2 leaky refine: grid honest **54276.18** → post full-train eval **10550.38** (exact match).
- Skrub-full run1: honest single model ~55k; raw median blend with leaky B ~43k; reported **42009.62**.

Run notes captured under `experiments/phase1/.../run*/notes.md`.

### 32) Evaluation tooling usage (phase-1)

```bash
# Per run
python test-scripts/analyze_run.py \
  --state experiments/phase1/california-housing-prices/skrub-full/gpt-5.4-mini/run2/final_state.json \
  --workspace experiments/phase1/california-housing-prices/skrub-full/gpt-5.4-mini/run2 \
  --json > experiments/phase1/.../run2/analysis.json

# Batch
python test-scripts/aggregate_runs.py --root experiments/phase1 --out experiments/phase1/results/phase1_all_runs.csv
```

**Still missing:** static leakage heuristics in `analyze_run.py` (e.g. flag `fit(X, y)` before holdout metric print, cross-split ensemble eval).

### Known open items (updated 2026-06-14)

- [ ] Fix vanilla `debug_prompt.py` on `vanilla-baseline` (or re-bootstrap with `--revert-prompts`).
- [ ] Add prompt/exec guards: never print `Final Validation Performance` after full-data fit used for that metric.
- [ ] Enable `use_data_leakage_checker=True` once checker covers patterns A–C.
- [x] Complete phase-1 grid (12 runs) — see §33–36 below.
- [ ] Tier-B holdout scoring (optional) under `experiments/phase1/eval/`.
- [x] Compare honest bands in `experiments/phase1/results/phase1_summary.md` (init / refine / ensemble; leakage exclusions documented).

### Files touched (2026-06-12 → 2026-06-14)

```
scripts/bootstrap_vanilla_baseline.sh
docs/experiment_plan_evaluation.md
test-scripts/analyze_run.py
test-scripts/aggregate_runs.py
experiments/phase1/california-housing-prices/   # archived run1/run2 × vanilla/skrub-full
docs/todo.md, docs/lessons.md
agents/.../ (P1–P7, context, FE-drop — see b7e4adf)
```

---

## TL;DR — from vanilla MLE-STAR baseline to now (2026-06-14)

**Vanilla baseline**

- Bootstrapped **`vanilla-baseline`** worktree (`mle-star_vanilla`) from **`ffa365c`** via `bootstrap_vanilla_baseline.sh` — same ChatAI/DDG/GPT-5 infra as improved branch, **no** skrub skill, TableReport, or tuning stage.
- Without `--revert-prompts`, **`debug_prompt.py` still nudged skrub** on package errors; init prompts stayed sklearn-only.

**Skrub-full prototype (improved branch)**

- Shipped **P1–P7**: CatBoost-safe **`choose_from` tuning**, bake integrity, search budget, ablation contract, anti-drift debug, context bloat fix, tuning must **preserve structural FE**.
- **`analyze_run.py` / `aggregate_runs.py`** + **`experiment_plan_evaluation.md`** for phase-1 protocol.

**Phase-1 California Housing (4/8 runs done)**

- **skrub-full run2** (~53k final, ~53k refine): **credible** — best honest comparison point for skrub.
- **skrub-full run1** (~42k final): **invalid** — HGB not CatBoost; ensemble cross-split + val calibration leakage.
- **vanilla run1** (~50k final): **usable** with mild ensemble optimism; refine didn't promote.
- **vanilla run2** (~10k refine/final): **invalid** — classic leak: train on full `X,y`, score `X_val`.

**Cross-cutting lesson**

- Parsed **`Final Validation Performance` is not trustworthy** without auditing the script; both systems can report fantasy RMSE when full-train refit or ensemble tricks meet the same holdout used for selection.
- For papers/comparison, report **init / honest refine / ensemble0** (or manual re-exec with fixed protocol), not submission-stage inflated scores.
- **Do not parallelize** multiple runs on the same task in one checkout (shared `workspace/`).

**Next** *(see also §33–36, 2026-06-15)*

- Backbone drift gate; leakage guards in `analyze_run.py`; fix vanilla debug prompt; selective post-P8 re-runs (not full 12-run grid until tune/refine stabilize).

---

## Progress update (2026-06-14 → 2026-06-15): full phase-1 grid, metric/tune fixes, results write-up

### 33) Spaceship Titanic task + phase-1 grid completed

- Added `tasks/spaceship-titanic/` (`task_description.txt`, Kaggle `train.csv` / `test.csv`); `config.py`: `lower=False`, classification.
- Ran **12 archived runs**: 2 tasks × `{vanilla, skrub-full}` × 3 repeats under `experiments/phase1/<task>/<system>/gpt-5.4-mini/run{1,2,3}/`.
- California housing extended from 4 → **6 runs** (added run3 both systems). Legacy `spaceship-titanic/skrub-full/.../legacy/run1` kept but excluded from main stats.

### 34) First Spaceship skrub-full runs — analysis (pre-fix runs)

- **Backbone drift:** CatBoost missing → debug/init swapped to HGB; `plan_implement` later emitted LogisticRegression / RF; anti-drift prompt alone insufficient.
- **Wrong tune metric:** tuning prompts/skills hardcoded RMSE → `mean_squared_error` on classification runs; tune search scores meaningless.
- **`tune_best_params` scramble:** positional `data_op__N` → plan name mapping assigned `max_depth=0.08`, `learning_rate=5` (Titanic run1).
- **TableReport:** profile (~1.2k chars) fed ablation agent on all skrub runs; useful for plan text, did not yield Titanic score gains.
- **Promotion:** housing skrub run2–3 refine promoted (+2.4k RMSE); Titanic `gain_refinement=0` all runs.

### 35) P8 — task-general metrics + value-aware `map_tuning_best_params` (shipped)

**No metric contract / no exec gates** (per design choice — prompt + skill only).

| Area | Change |
|------|--------|
| `tuning/prompt.py`, `refinement/prompt.py`, `submission/prompt.py` | “Competition metric from task description” instead of RMSE mandates |
| Skill docs (`SKILL.md`, `dataops_api_quickmap.md`, `choices_hparam_pattern.md`, `common_failure_fixes.md`, …) | Generic holdout score wording; RMSE only as regression example |
| `code_util.map_tuning_best_params` | **Value-aware** match to plan `kind` / ranges (fixes `data_op__` order ≠ plan order) |
| `tune_implement` prompt | Build human-named `TUNING_BEST_PARAMS`, not raw `data_op__N` keys |
| `tests/test_code_util.py` | Titanic scramble case + identity + `choose_from` |

**Not yet shipped:** deterministic **backbone drift gate** in `debug_util` / `code_util` (planned: extended estimator regex + set-equality check on debug / plan_implement / tune).

### 36) Phase-1 results aggregation + `phase1_summary.md`

Generated under `experiments/phase1/`:

| Artifact | Role |
|----------|------|
| `manifest.csv` | 12-run index (paths, scores, debug, wall/exec) |
| `results/phase1_all_runs_12.csv` | `analyze_run` rows (primary) |
| `results/phase1_by_task_system.csv` | Means per (task, system) |
| `results/phase1_summary.md` | Full write-up + **TL;DR** |
| `README.md` | Layout + regenerate commands |

**Analysis protocol (updated):**

- Exclude probable leakage for score claims: **housing skrub run1**, **vanilla run2**; all Titanic runs kept.
- Report **val scores by stage** (init → refine → tune → ensemble → submission).
- Report **Python exec** (sum of scored `subprocess` times from `final_state.json`) **and wall time** (ADK log start→end), **Python fraction**, **exec runs per stage**, **`tune_winner_source`**, refinement promotion (`gain_refinement` / `gain_tuning`).
- Timing definitions documented in summary (exec ≠ wall; exec includes retries/ablation scripts).

**Headline results (clean runs, operational hypothesis):**

| Signal | Housing (n=2) | Titanic (n=3) |
|--------|-----------------|-----------------|
| Val scores | ~tie (~52–56k RMSE) | vanilla **+2.5 pp** accuracy |
| Python exec | skrub **~2.6× faster** | skrub **~42% faster** |
| Wall time | skrub **~2.3× shorter** | skrub **~18% shorter** |
| DataOps adherence | **~0.71** vs 0 | **~0.64** vs 0 |
| Fewer debug rounds | **No** (more skrub refine debug) | **No** |
| Tuning beat structural | — | **No** (`tune_winner_source=structural` all runs) |

**Credit planning (estimate, not logged):** ~€2–3/skurb-full run, ~€0.45/vanilla; full 12-run grid ≈ €40–65; €25/mo budget → ~8–12 skrub runs or selective re-runs after P8/backbone fixes.

### Known open items (updated 2026-06-15)

- [x] Complete phase-1 grid (12 runs) + `phase1_summary.md`.
- [x] P8 metric generalization + `map_tuning_best_params` value-aware fix + tests.
- [ ] **Backbone drift gate** (extended regex + post-edit check in `get_code_from_response`).
- [ ] Fix vanilla `debug_prompt.py` (or re-bootstrap with `--revert-prompts`).
- [ ] Prompt/exec guards: no holdout metric after full-data fit (patterns A–C).
- [ ] Enable `use_data_leakage_checker=True` once checker covers A–C.
- [ ] Optional: log LiteLLM token usage into `meta.json`; static leakage flags in `analyze_run.py`.
- [ ] Selective re-runs (post P8 + backbone gate) — not full 12-run grid until tune/refine stabilize.

### Files touched (2026-06-14 → 2026-06-15)

```
agents/.../sub_agents/{tuning,refinement,submission}/prompt.py
agents/.../shared_libraries/code_util.py          # map_tuning_best_params value match
agents/.../skills/skrub-dataops-pipeline/       # metric-general wording
agents/.../tests/test_code_util.py
experiments/phase1/                               # 12-run archive + manifest
experiments/phase1/results/phase1_summary.md
experiments/phase1/README.md
```

### TL;DR (2026-06-15 addendum)

- **Phase-1 done:** 12 runs archived; compared vanilla vs skrub-full on housing + Spaceship Titanic (3 repeats each).
- **Scores:** housing tie on honest band; **vanilla wins Titanic**; exclude known leaky housing runs from claims.
- **Skrub operational win:** faster Python + wall time, **~65% DataOps adherence**; **not** fewer debug rounds; **tuning never beat structural**.
- **Shipped P8:** task-agnostic metric prompts + fixed `tune_best_params` mapping; **next:** backbone drift gate, then selective re-runs.

### 37) `analyze_run.py` — phase-1 reporting fields

Extended (or first used at scale) for archived run tables — not repeated in §32 tooling blurb:

| Field / struct | Source | Used for |
|----------------|--------|----------|
| `stage_timing[stage].exec_seconds` | Sum `execution_time` on scored `*_exec_result_*` keys | Per-stage Python time |
| `stage_timing[stage].exec_runs` | Count of those keys per stage | Retry/churn proxy |
| `extras.log_wall_seconds` | ADK log `Script started on` → `Script done on` | Full session wall time |
| `to_row()` gains | init vs refine/tune promoted | `gain_refinement`, `gain_tuning` |
| `tune_winner_source` | `final_state.json` | Whether tune beat structural |

**Note:** Python exec counts only **scored** subprocess runs (includes ablation/inner-loop retries); wall time includes all LLM/tools/waiting.

### 38) `phase1_summary.md` — reporting revisions (post-aggregation)

After initial 12-run aggregation, summary was revised in passes (without rewriting score/debug tables):

1. **Leakage exclusions** for score claims: housing skrub `run1`, vanilla `run2`; Titanic all kept.
2. **Per-stage val scores** (init → refine → tune → ensemble → submission) + per-stage Python exec + debug rounds.
3. **Timing definitions** block: val score, Python exec, wall time, Python fraction, exec runs, debug — with derivation notes.
4. **Run-level timing** per task: wall vs total Python exec side-by-side; Python fraction; exec runs per stage.
5. **Promotion & tuning:** `gain_refinement` / `gain_tuning`; `tune_winner_source=structural` on all skrub runs.
6. **Conclusion reframed** around **operational hypothesis** (faster exec, DataOps adherence, fewer debugs) vs score hope.
7. **TL;DR appended** at file end: vanilla vs skrub, **TableReport** + **terminal tuning** novelty assessment, skrub benefits, bottom line.

See `experiments/phase1/results/phase1_summary.md` (full tables + TL;DR).

### 39) Backbone drift gate — design agreed, not shipped

From Spaceship Titanic + housing phase-1 post-mortem (prompt-only anti-drift insufficient):

| Gap | Detail |
|-----|--------|
| Regex | `\b([A-Z]\w*(?:Regressor\|Classifier))\b` misses `LogisticRegression`, `Ridge`, `SVC`, … → empty debug contract |
| Enforcement | No exec gate on drift; promotion saved scores but wasted debug/exec |

**Planned surgical fix (~50 LOC, deferred after P8):**

- `extract_estimator_classes()` + `check_backbone_unchanged()` in `code_util.py` (extended regex)
- Hook in `debug_util.get_code_from_response` → synthetic failure before `evaluate_code`
- Check: `*debug*`, `plan_implement*`, `tune_implement` / `tune_bake` vs structural baseline
- Skip: `model_eval`, `merger`, `ensemble_*`, `submission`

### 40) API credit planning (estimate from archived runs)

No token logging in `meta.json` today; estimated from `LiteLLM completion()` counts in ADK logs + list pricing for `gpt-5.4-mini`:

| System | LLM calls/run (observed) | Planning €/run |
|--------|--------------------------|----------------|
| vanilla | ~22–36 | ~€0.35–0.55 |
| skrub-full | ~86–179 (avg ~126) | ~€2.00–2.80 typical; up to ~€5 heavy debug |

**€25/month:** full 10 tasks × 2 variants × 2–3 runs (40–60 runs) ≈ **€58–73** — not feasible at typical rates. Prefer **selective re-runs** post P8 + backbone gate (~8–12 skrub runs or ~10 tasks × 1 run × both variants). Optional later: log LiteLLM `usage` into `meta.json`.

### Known open items (updated 2026-06-15, continued)

- [x] Phase-1 summary reporting protocol (leakage exclusions, wall/exec/fraction, promotion, TL;DR) — §38.
- [x] Document backbone drift gate design — §39; **implement** still pending.
- [x] Credit/run budget estimate for scaled eval — §40.

### Files touched (2026-06-15, continued)

```
test-scripts/analyze_run.py                    # stage_timing, wall, gains, tune_winner (phase-1 tables)
experiments/phase1/results/phase1_summary.md   # revised reporting + TL;DR (multiple passes)
```

---

## Progress update (2026-06-28): runtime contracts — backbone drift gate, tuning search, ablation

Shipped deterministic pre-exec gates in `code_util.py` + stable backbone contract in `debug_util.py`. Supersedes §39 design (which planned broader enforcement on `plan_implement` / `tune_bake`); actual scope is **init + tune search only** — refinement may legitimately swap backbones per plan.

### 41) Backbone drift gate (debug anti-drift fingerprint)

**Problem:** Phase-1 runs showed debug agents swapping model families (CatBoost → HGB/RF/Ridge) to silence errors. Prompt-only `_get_backbone_contract(code)` made it worse: it read estimators from the **current buggy code**, so the anchor moved with every failed fix.

**Fix:** Pre-exec gate + **stable anchors** (not derived from drifting code).

| Function | Role |
|----------|------|
| `_estimator_classes()` | Regex `\b([A-Z]\w*(?:Regressor\|Classifier))\b` on code text |
| `retriever_estimator_classes()` | Parse classes from retriever `model_name` |
| `resolve_backbone_required()` | Resolve required set + label for error messages |
| `backbone_drift_violation()` | Set-equality check; returns concise `stderr` or `None` |
| `should_enforce_backbone_drift()` | `True` only for `model_eval*` and `tune_implement*` |
| `preexec_code_failure()` | Compile → backbone → (tune only) search contract, before subprocess |
| `maybe_set_debug_anchor()` | Snapshot first failing init script when retriever name has no parseable class |

**Anchors (what “required backbone” means):**

| Stage | Required estimator set from | State keys |
|-------|----------------------------|------------|
| Init / init debug | Retriever model name | `init_{task}_model_{id}.model_name` |
| Init fallback | First failing init script (if retriever unparsable) | `debug_anchor_code_{task_id}_{model_id}` |
| Tune / tune debug | Structural solution | `train_code_{outer_loop_round}_{task_id}` |

**Not enforced:** `plan_implement*` (refinement may swap backbone if plan says so), `merger*`, `ensemble_*`, `submission`, `tune_bake`, `ablation*`.

#### How debug drift checking works (end-to-end)

```
Agent outputs code
  → evaluate_code()
      → preexec_code_failure()          # cheap, no subprocess
          1. compile(raw_code)
          2. if model_eval* or tune_implement*:
               required, label = resolve_backbone_required(...)
               backbone_drift_violation(required, raw_code, label=label)
          3. if tune_implement*: tune_search_contract_violation(raw_code)
      → on failure: store {returncode:1, stderr:"Backbone drift: ..."}
      → on pass + run gates: subprocess run

Debug loop (on failure):
  bug_summary_agent  ← stderr (trimmed)
  debug_agent        ← BUG_REFINE_INSTR + {bug} + {backbone_contract}
                       backbone_contract from resolve_backbone_required()
                       (same anchor as pre-exec — NOT from buggy code)
  debug output       → evaluate_code() again (pre-exec re-checks drift)
```

**Example rejection (`stderr`):**

```
Backbone drift: keep estimator classes ['CatBoostClassifier'] (retriever: CatBoostClassifier); got ['RandomForestClassifier']. Fix the reported error without swapping model families or simplifying to a different pipeline.
```

**Why this stops drift:** The debug agent sees the **same stable required set** in both the prompt (`# Backbone contract`) and the execution failure message. Swapping families fails pre-exec immediately (0 s subprocess) until the fix restores the required class set.

**Files:** `shared_libraries/code_util.py`, `shared_libraries/debug_util.py`, `tests/test_code_util.py`.

### 42) Tuning search contract (pre-exec)

Separate from backbone drift; enforced only on `tune_implement*` inside `preexec_code_failure()`.

| Required in tune search script | Purpose |
|-------------------------------|---------|
| `choose_*` | In-graph hyperparameter nodes |
| `make_randomized_search` | Real search, not fixed-params fake tune |
| `search.fit` | Search executed on holdout bind |
| `TUNING_BEST_PARAMS` print | Handoff to bake / state mapping |

**Failure message:**

```
Tuning search contract: missing choose_*, make_randomized_search, ... Fix the error without removing the search block or choose_* nodes.
```

Also enforced at run time (post-subprocess): exit 0 without parseable `TUNING_BEST_PARAMS` JSON → synthetic failure (`evaluate_code`). Bake stage: `code_contains_tuning_placeholders()` gate in `get_run_code_condition` (no `choose_*` in baked script).

**Note:** `tune_structural_fingerprint_violation()` was removed — tuning backbone checks use `backbone_drift_violation()` via `resolve_backbone_required()` directly (no wrapper).

### 43) Ablation contract (post-exec)

Enforced in `evaluate_code()` after subprocess for `ablation*` only.

| Rule | Mechanism |
|------|-----------|
| ≥2 `Ablation[...]` lines in stdout | Baseline + at least one variant |
| On violation | `returncode=1`, stderr names contract; debug preserves variants + backbone |

Prompt/skill alignment: `references/feature_engineering_skrub.md`, `references/ablation_dataops_template.md` — baseline variant must match input backbone unless model swap is the explicit hypothesis.

### 44) Estimator regex gap (models without Classifier/Regressor suffix)

If retriever `model_name` and code use estimators like `LogisticRegression`, `Ridge`, `SVC`, `XGBRegressor` (no `Classifier`/`Regressor` suffix), `_estimator_classes()` returns **empty** → `required` is empty → **backbone drift check is skipped** (no crash). Init fallback anchor may still capture classes from the first failing script if that script happens to use `*Classifier`/`*Regressor` names. This is the known gap noted in §39; extending the regex is still open.

### Known open items (updated 2026-06-28)

- [x] Backbone drift gate for init + tune search (§41).
- [ ] Extend estimator regex for `LogisticRegression`, `Ridge`, `SVC`, etc. (§44).
- [ ] Plan-aware backbone check for refinement (when plan explicitly swaps model).
- [ ] Fix vanilla `debug_prompt.py` (or re-bootstrap with `--revert-prompts`).
- [ ] Prompt/exec guards + leakage checker for patterns A–C.

### Files touched (2026-06-28)

```
agents/.../shared_libraries/
  code_util.py       # preexec_code_failure, backbone_drift_violation, tune_search_contract_violation
  debug_util.py      # _get_backbone_contract → resolve_backbone_required (stable anchor)

agents/.../tests/test_code_util.py
docs/WORKING_PROGRESS_MLE_STAR_SKRUB.md
```

### TL;DR (2026-06-28)

- **Backbone drift:** pre-exec set-equality on estimator classes; enforced init + tune search only; debug prompt uses retriever/structural anchor, not buggy code.
- **Tune search contract:** pre-exec requires `choose_*` + search + `TUNING_BEST_PARAMS`; bake still gated separately.
- **Ablation contract:** post-exec stdout line count; unchanged from P7.
- **Gap:** no suffix match → drift gate disabled, pipeline does not break.

