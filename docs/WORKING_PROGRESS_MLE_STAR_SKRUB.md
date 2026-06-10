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

**`SKILL.md`** — starter template rewritten to the **two-block holdout pattern** (Block 1: `train_part` for metric; Block 2: `train_df` for test/submission). Rule #13 enforces honest holdout binding.

**Prompt alignment (minimal one-liners, no bloat):**
- `sub_agents/initialization/prompt.py` — load quickmap + two-block holdout
- `sub_agents/refinement/prompt.py` — ablation/implement fit on `train_part` only
- `sub_agents/tuning/prompt.py` — bake scores via Block 1, not full-train fit
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

**Correct pattern (what agents should emit after rerun):**

```python
# Block 1 — metric
data_train = skrub.var("data", train_part)
# ... build graph on data_train ...
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
print(f"Final Validation Performance: {rmse}")

# Block 2 — test/submission (after metric print)
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
| 4 | Optimistic validation RMSE (skrub bind) | Skill docs two-block holdout + prompt one-liners |
| 5 | Ablation RMSE ≠ solution RMSE | Ablation contract + FE reference; holdout bind alignment (ongoing) |
| 6 | `final_state.json` bloated by tune stdout | Still open — truncate before persist (see future improvements) |

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

test-scripts/analyze_run.py
docs/WORKING_PROGRESS_MLE_STAR_SKRUB.md        # this file
```

### Known open items (updated)

- Re-run California Housing (or new task) **after holdout skill update**; confirm Block 1 in generated scripts and rising-but-honest RMSE.
- Confirm `map_tuning_best_params` in `final_state.json` on fresh run (`r4` snapshot may still show `data_op__*` if captured before fix).
- Enable `use_data_leakage_checker=True` once holdout docs stabilize; monitor checker doesn’t block on tool-only turns.
- Truncate tune search stdout before state write; feed TableReport to planners (not only ablation).
- Unified validation harness / same-pipeline ablation still desirable for planner trust.

### TL;DR (current prototype state)

- **MLE-STAR + skrub:** OpenAI/ChatAI-compatible runtime, native ADK `skrub-dataops-pipeline` skill, structural refinement with TableReport-guided ablation, and a **separate terminal tuning stage** (`choose_*` → search → bake → promote).
- **Main correctness push:** holdout data leakage from `skrub.var("data", train_df)` + early `make_learner(fitted=True)` — fixed in skill references and light prompt patches; optional leakage checker loads `holdout_data_leakage.md`.
- **Tuning hardening:** JSON-safe best params, `data_op__N` → plan name mapping, bake placeholder gate, promotion from **`train_tune_baked.py`** only.
- **Results:** `r4` tuning promoted (`~23k` vs structural `~24k` RMSE) but cross-stage scores were misleading pre-holdout fix; re-run needed for trustworthy comparison.
- **Next:** honest holdout rerun, enable leakage checker optionally, stdout/state slimming, planner+ablation pipeline alignment.

