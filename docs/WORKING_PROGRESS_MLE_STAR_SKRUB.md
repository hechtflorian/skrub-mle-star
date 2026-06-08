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

