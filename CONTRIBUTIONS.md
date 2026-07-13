# Contributions

What was built on top of upstream MLE-STAR ([Google ADK samples](https://github.com/google/adk-samples), `python/agents/machine-learning-engineering`, Apache-2.0), grouped by what we set out to do, with clickable pointers to the files we wrote or changed.

## Baselines for comparison

- **Upstream** = the original Google ADK sample (our initial import).
- **Vanilla** = branch `vanilla-baseline`, sibling worktree `[../mle-star_vanilla/](../mle-star_vanilla/)` (next to the `skrub-mle-star/` checkout). This is upstream **plus our OpenAI/ChatAI runtime-compatibility layer**, so those runtime changes are still **our** contribution even though they also live in vanilla.
- **Improved** = branch `main` in `skrub-mle-star/` (this tree), everything in vanilla **plus** the skrub DataOps skill, TableReport profiling, the tuning stage, drift/robustness guards, and prompt hardening.
- Some sections might dublicate, if modifications are wired through multiple files. We group by MLE-STARs folder structure: `shared_libraries`, `sub_agents`. `skills`, etc.

```bash
# What we changed vs the untouched upstream sample
git diff 6c96e03..HEAD -- agents/machine-learning-engineering/machine_learning_engineering

# What is improved-only (skrub work) vs our vanilla baseline
git diff vanilla-baseline..HEAD -- agents/machine-learning-engineering/machine_learning_engineering
```

---

## Team


| Member            | Git author(s)                | Contributions                                                                                                                                                                                                                                 |
| ----------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Florian Hecht** | `fhecht`                     | Runtime compat, skrub skill wiring + references, Tablereport novelty, Tuning stage novelty, agent drift/robustness guards, prompts, experiment harness refinement, experiment execution, analysis & experimental results writeup, docs, tasks |
| **Yuquan Cui**    | `Yuquan Cui`                 | Base automated-experiment harness (`evaluate.py`, task manifest, experiment doc)                                                                                                                                                              |
| **Xiaomei Long**  | `Xiaomei Long`, `Xiaomei`    | Initial `skrub-dataops-pipeline` skill package + first ADK skill wiring; initial deterministic tuning template; tune-bake drift guards                                                                                                        |
| **Zhengkun Wang** | `wang.zk25`, `Zhengkun Wang` | Repository initial import / setup                                                                                                                                                                                                             |


### What we did **not** write

- Core MLE-STAR multi-agent architecture, ADK agent definitions, and most orchestration; **upstream**.

---

# Shared libraries `shared_libraries/`

Path prefix for this section: `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/`.

## 1. OpenAI / ChatAI runtime compatibility (shared with vanilla)

**Goal:** run MLE-STAR on OpenAI-compatible providers (ChatAI/OpenAI, via LiteLLM) while keeping Gemini working. These are in both `vanilla-baseline` and `main` (improved skrub-full).


| What                                                                                           | Pointer                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model-aware **web search**: DuckDuckGo (`ddgs`) for non-Gemini, ADK `google_search` for Gemini | `[search_tool_util.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/search_tool_util.py)` - `ddg_web_search`, `get_search_tools` (new file)        |
| **GPT-5 temperature guard** (LiteLLM rejects non-`1.0` temp for GPT-5) + config flags          | `[config.py#L50-L67](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/config.py#L50-L67)` - `is_gpt5_family_model`, `get_compatible_temperature`       |
| **Response parsing** for openai-providers that return `text=None` parts                        | `[common_util.py#L12-L34](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/common_util.py#L12-L34)` - `get_text_from_response` (skips non-`str` parts) |


`get_compatible_temperature` is wired at every LLM call site (`agent.py`, all sub-agents, `debug_util.py`, `check_leakage_util.py`).

To use openai-compatible models, only the prefix "openai" ( `openai/<model>` ) and API-credentials in `.env` must be setup (see `.env.example`). The prefix ensures that Litellm is used for model routing.

Author: Florian Hecht

## 2. Skrub DataOps skill infrastructure (improved-only)

**Goal:** give code-writing and planning agents an on-demand ADK skill so generated code uses skrub DataOps instead of ad-hoc sklearn, without bloating every prompt or unnessary complexity, while skill references can be updated/appended easily in the future.


| What                                                                                                                  | Pointer                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ADK SkillToolset wrapper** + per-call logging + combined skill/search tools for search-enabled agents (init, debug) | `[skill_tool_util.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/skill_tool_util.py)` - `get_skill_toolset` (L103), `get_skill_and_search_tools` (L111) (new file) |
| Leakage checker **skill-enabled** + made robust to empty tool-call turns                                              | `[check_leakage_util.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py)` - see below                                                            |
| Leakage prompts require loading the leakage reference; tool calls are prep-only and agents must finish real response  | `[data_leakage_prompt.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/data_leakage_prompt.py)` - `CHECK_LEAKAGE_INSTR` (L3), `LEAKAGE_REFINE_INSTR` (L24)           |


`skill_tool_util` is wired into the leakage + debug utilities and every planning/code-writing sub-agent where necessary: `[debug_util.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py)`, `[check_leakage_util.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py)`, and `sub_agents/{initialization,refinement,tuning,ensemble,submission}/agent.py`.

`check_leakage_util.py` specifics (our changes vs upstream):

- Skill toolset attached to both checker + refiner agents ([L207](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py#L207), [L244](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py#L244)).
- `parse_leakage_status` guards against missing JSON before `json.loads` ([L49-L62](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py#L49-L62)).
- `update_extract_status` initializes `leakage_status/code_block/extract_status` up front and only parses when `response_text.strip()` is non-empty, so empty skill-tool turns no longer corrupts state or waste agent turns ([L63-L100](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py#L63-L100)).
- `replace_leakage_code` returns early on empty response ([L145](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/check_leakage_util.py#L145)).

Generally, skill-enabled agents responses need to be parsed while checking if their response may be empty (skill-call only, no real response in same turn).

Author: Florian Hecht

## 3. TableReport data profiling for targeted refinement (improved-only)

**Goal:** give the refinement/ablation planner a compact, structured view of the dataset.


| What                                                                           | Pointer                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `skrub.TableReport` → compact ablation profile; state accessor with column cap | `[table_report_util.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/table_report_util.py)` - `load_table_report_dict` (L39), `format_ablation_profile` (L95), `get_profile_from_state` (L24) (new file) |


Wired into the refinement outer loop in `[sub_agents/refinement/agent.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py)` (builds `ablation_table_report_profile`_* and writes `workspace/<task>/table_report.json`).

Example of the compact TableReport profile fed into the ablation agent context (`{data_profile}`, abalone-regression):

```text
Dataset: 90,615 rows · 10 columns (0 constant)
Composition: 9 numeric · 1 string/categorical · 0 columns with missing values

Columns:
  column           dtype        nulls   unique   high_card
  ---------------  -----------  ------  -------  ---------
  id               Int64        0.0%     90,615  yes
  Sex              Object       0.0%          3  no
  Length           Float64      0.0%        157  yes
  Diameter         Float64      0.0%        126  yes
  Height           Float64      0.0%         90  yes
  Whole weight     Float64      0.0%      3,175  yes
  Whole weight.1   Float64      0.0%      1,799  yes
  Whole weight.2   Float64      0.0%        979  yes
  Shell weight     Float64      0.0%      1,129  yes
  Rings            Int64        0.0%         28  no

Top associations (Pearson):
  Length          <-> Diameter          0.99
  Whole weight    <-> Whole weight.2     0.98
  Whole weight    <-> Whole weight.1     0.97
  Whole weight    <-> Shell weight       0.96
  Whole weight.1  <-> Whole weight.2     0.95
  Whole weight.2  <-> Shell weight       0.94
  Diameter        <-> Height             0.93
  Diameter        <-> Whole weight       0.93
```

Author: Florian Hecht

## 4. Backbone-drift guards & debug contracts (improved-only)

**Goal:** stop debug/ablation/plan agents from silently swapping the optimal backbone model family (given by retriever) or dropping DataOps blocks just to make an error disappear. Design: `[docs/BACKBONE_DRIFT_GUARDS.md](docs/BACKBONE_DRIFT_GUARDS.md)`.

`code_util.py` : estimator/anchor detection + violation checks:


| Function                                                                         | Line                                                                                                                  | Purpose                                               |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `_estimator_classes` / `canonical_estimator_name` / `canonical_estimator_set`    | [L23-L45](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L23-L45)     | Extract & normalize estimator classes from code       |
| `retriever_estimator_classes`                                                    | [L47-L53](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L47-L53)     | Anchor set from init retriever                        |
| `backbone_check_mode` / `should_snapshot_debug_anchor` / `debug_anchor_code_key` | [L54-L73](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L54-L73)     | Per-agent drift-check policy                          |
| `resolve_backbone_required` / `backbone_violation`                               | [L74-L162](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L74-L162)   | Compute required anchor & detect a swap               |
| `maybe_set_debug_anchor`                                                         | [L208-L223](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L208-L223) | Snapshot structural anchor before debugging           |
| `preexec_code_failure`                                                           | [L224-L289](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L224-L289) | Pre-execution reject with a fix hint (drift/contract) |


`debug_util.py` : inject anchor contract into the debug prompt:

- `_get_backbone_contract` builds the contract text ([L159](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py#L159)) and is wired into `get_debug_agent_instruction` ([L232](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py#L232)).

`debug_prompt.py` : `BUG_REFINE_INSTR` reworked ([L10-L40](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_prompt.py#L10-L40)): smallest-possible-change rule, preserve DataOps structure & honest holdout, skill-load only for skrub errors, and a `{backbone_contract}` slot.

Some backbone contracts are strict (e.g. must match exact set for `init`, `tuning`) while some only need to keep at least one backbone (e.g. `ablation`, `refinement` should be able to explore). Pre-exec gates are cheap so that false scripts don't run, while we intentionally kept them as light as possible to not hinder agent exploration while still meeting our requirements; nudging in either direction can be a tradeoff.

Author: Florian Hecht, Xiamei Long

## 5. Execution-robustness gates (improved-only)

**Goal:** never treat an empty/tool-call-only turn as valid code, keep state small, and parse validation scores defensively.

`code_util.py`:

- `get_run_code_condition` - for `ablation` / `plan_implement` / `ensemble_plan_implement` / `tune`_*, require non-empty **and** `compile()`-valid Python before running ([L657-L749](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L657-L749)).
- `evaluate_code` : routes through `preexec_code_failure`, enforces the ablation "≥2 Ablation[...] lines" contract, and the tuning `TUNING_BEST_PARAMS` contract ([L750+](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L750)).
- `truncate_for_state` - cap huge train stdout (e.g. for verbose models) before it is stored in state / `final_state.json` ([L329-L348](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L329-L348)).
- `extract_performance_from_text`-  hardened validation-score parsing to avoid reading empty scores and crashes ([L527-L545](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L527-L545)).

`debug_util.py`:

- `get_code_from_response` : early-return on empty code and on empty extracted block (an empty `str.replace("")` would explode `prev_code` into a multi-MB string and flood agent context window) ([L236-L288](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py#L236-L288)).
- `get_debug_inner_loop_agent` now attaches skill+search tools ([L331](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py#L331)); `get_run_and_debug_agent` takes an optional `tools=` and skips the leakage checker for `submission` to avoid drift ([L369+](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py#L369)).

`common_util.py`: the `text=None` guard in [§1](#1-openai--chatai-runtime-compatibility-shared-with-vanilla) also protects skill-tool-only turns.

Author: Florian Hecht

## 6. Tuning stage plumbing (improved-only)

**Goal:** support a dedicated terminal `choose`_* search/bake stage (the stage agents live in `sub_agents/tuning/`, see [§8](#8-still-to-detail)).

`config.py`: `tuning_enabled`, `tuning_n_iter`, `table_report_enabled` flags ([L41-L43](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/config.py#L41-L43)).

`code_util.py` : tuning param handling & state keys:


| Function                                                                                                                      | Line                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `normalize_tuning_best_params`, `extract_tuning_best_params`                                                                  | [L349-L375](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L349-L375) |
| `_data_op_sort_key`, `_normalize_param_value`, `_param_match_cost`, `_map_params_by_value_match`, `_coerce_tuned_param_value` | [L376-L482](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L376-L482) |
| `map_tuning_best_params` (map generic `data_op__N` → plan param names)                                                        | [L483-L517](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L483-L517) |
| `code_contains_tuning_placeholders` (bake must not still contain `choose`_*)                                                  | [L518-L526](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/code_util.py#L518-L526) |


Plus `tune_implement`/`tune_bake` branches added to `get_updated_suffix`, `get_code_state_key`, `get_code_execution_result_state_key`, and `evaluate_code`.

Author: Florian Hecht, Xiaomei Long

---

# Sub-agents & pipeline — `sub_agents/` and `agent.py`

Path prefix for this section: `agents/machine-learning-engineering/machine_learning_engineering/`.

## 7. Pipeline wiring (`agent.py`)

**Goal:** insert the new tuning stage between refinement and ensemble, gated by config.

- Import tuning module + build `_pipeline_sub_agents` conditionally on `config.CONFIG.tuning_enabled` (`[agent.py#L38-L68](agents/machine-learning-engineering/machine_learning_engineering/agent.py#L38-L68)`). Vanilla pipeline is a fixed `init → refine → ensemble → submission`; ours becomes `init → refine → [tune] → ensemble → submission`.

Author: Florian Hecht, Xiaomei Long

## 8. Skill-tool wiring across code-writing agents

**Goal:** every agent that writes/edits code can call the skrub DataOps skill on demand. Pattern: attach `skill_tool_util.get_skill_toolset()` (or `get_skill_and_search_tools()` where web search is also needed). This replaces the vanilla `search_tool_util`-only wiring.


| Agent                                               | Pointer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Notes                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| Model retriever                                     | `[initialization/agent.py#L413](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/agent.py#L413)`                                                                                                                                                                                                                                                                                                                                                                                                                                                 | skill **+** search (was `search_tool_util.get_search_tools`) |
| Model-eval / merger / data-use checkers             | `[initialization/agent.py#L441](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/agent.py#L441)`, [L457](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/agent.py#L457), [L483](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/agent.py#L483)                                                                                                                                                                                                             | skill toolset                                                |
| Ablation / init-plan / plan-implement / plan-refine | `[refinement/agent.py#L427](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L427)`, [L490](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L490), [L515](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L515), [L522](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L522), [L537](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L537) | skill toolset                                                |
| Ensemble init-plan / implement / refine             | `[ensemble/agent.py#L211](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L211)`, [L225](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L225), [L233](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L233), [L247](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L247)                                                                                                                         | skill toolset                                                |
| Submission                                          | `[submission/agent.py#L75](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/submission/agent.py#L75)`                                                                                                                                                                                                                                                                                                                                                                                                                                                           | skill toolset                                                |
| Debug inner loop (shared)                           | `[shared_libraries/debug_util.py#L331](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py#L331)`                                                                                                                                                                                                                                                                                                                                                                                                                                              | skill **+** search                                           |


Author: Florian Hecht

## 9. Prompt hardening for skrub DataOps (all stages)

**Goal:** require DataOps-first pipelines, on-demand skill loading, honest holdout, and stage boundaries, without prompt bloat. Every prompt below adds: "load skill before uncertain edits", "tool calls are preparation only, finish with real response in the same turn", and DataOps-pipeline requirements.


| Prompt         | Pointer                                                                                                                                 | Key additions                                                                                                                                        |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Initialization | `[initialization/prompt.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/prompt.py)`      | `list_skills`→`load_skill`→`load_skill_resource`; DataOps-first; holdout-only early stages (speedup runtime); invalid if sklearn-only                |
| Refinement     | `[refinement/prompt.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/prompt.py)`              | `{data_profile}` slot; load `ablation_dataops_template.md`; per-variant `train_part` holdout; prefer FE/encoding before model swap or large searches |
| Ensemble       | `[ensemble/prompt.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/prompt.py)`                  | keep each solution's DataOps intact; load `ensemble_dataops_patterns.md` (Pattern A); holdout-only, strip test/submission blocks                     |
| Submission     | `[submission/prompt.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/submission/prompt.py)`              | load `submission_export.md`; full-train fit + test export only; no second holdout print                                                              |
| Debug (shared) | `[shared_libraries/debug_prompt.py](agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_prompt.py)` | see [§4](#4-backbone-drift-guards--debug-contracts-improved-only)                                                                                    |


**Stage templates (ablation + ensemble).** Beyond generic skill references, we gave the two most error-prone stages a **dedicated copy-the-skeleton template** so the agent starts from a valid DataOps shape instead of improvising. This both improves DataOps pipeline creation and cuts downstream debug iterations:

- Ablation → `[references/ablation_dataops_template.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/ablation_dataops_template.md)` (minimal DataOps ablation shape + variant patterns + stdout contract), loaded every ablation step.
- Ensemble → `[references/ensemble_dataops_patterns.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/ensemble_dataops_patterns.md)` (Pattern A multi-leg blend skeleton keeping each leg's pred chain intact), loaded for ensemble implement.

**Deferred submission export (efficiency trade-off).** In vanilla, early-stage scripts tend to also refit on the full `train_df`, load `test_df`, and write `submission.csv`. In improved we **nudge every early stage (init, ablation, refinement, tuning, ensemble) to stop after the holdout print** (no full retrain, no test load, no export) and **defer the full-train refit +** `test_df` **predict +** `./final/submission.csv` **export to the submission agent only**:

- Prompts + skill rules enforce "holdout metric only" upstream and reserve export for submission; see the initialization/refinement/ensemble/submission rows in [§9](#9-prompt-hardening-for-skrub-dataops-all-stages) and the honest-holdout rules in `[SKILL.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/SKILL.md)`.
- The submission agent gets a **dedicated reference it must load**, `[references/submission_export.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/submission_export.md)` (full-train refit, `test_df` predict, `./final/submission.csv`), wired via the submission prompt (`[submission/prompt.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/submission/prompt.py)`).
- **Goal:** cut redundant full-train/export work in every early stage (efficiency). **Known caveat:** the submission agent still debugs, and its debug loop can occasionally drift the pipeline (e.g. swapping the model family) so the exported submission diverges from the promoted structural/ensemble winner. It only happens sometimes, and the [backbone-drift guards](#4-backbone-drift-guards--debug-contracts-improved-only) reduce it, but it is a real risk we accept in exchange for the hoped-for efficiency gain.

Author: Florian Hecht, Xiaomei Long

## 10. Agent-level guards (empty tool-call / promotion correctness)

**Goal:** with skill tools attached, agents can emit tool-call-only turns (no code). These guards stop empty/duplicate turns from being scored as valid progress, and fix promotion logic.

**Refinement** (`[refinement/agent.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py)`):

- `check_plan_implement_finish`: finish only when exec `returncode==0`, `score` present, code non-empty **and** differs from previous ([L321](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L321)); also skips implement if no valid extracted block.
- `get_refined_plan` : ignore empty responses ([L403](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L403)).
- `update_outer_loop_states` : robust promotion: skips entries missing `score`, scans `improve_0..N`, keeps previous solution when nothing beats it ([L33](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L33)).

**Ensemble** (`[ensemble/agent.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py)`):

- `get_init_ensemble_plan` / `get_refined_ensemble_plan` :  ignore empty tool-call turns ([L38](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L38), [L71](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L71)).
- `check_ensemble_plan_implement_finish` :  require `returncode==0` **and** `score` ([L51](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L51)).
- `get_ensemble_plan_refinement_instruction` :  only rank plans that actually have a `score` (no blind `exec_result["score"]` access to avoid crashes) ([L102](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/ensemble/agent.py#L102)).

**Submission** (`[submission/agent.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/submission/agent.py)`): leakage-check skip disabled so submission code is audited if enabled ([L17-L20](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/submission/agent.py#L17-L20)).

Author: Florian Hecht

## 11. TableReport profiling (#2)

**Goal:** feed the ablation/plan agents a compact dataset profile (see [§3](#3-tablereport-data-profiling-improved-only) for the util).

- `init_outer_loop_states` builds the profile and writes `table_report.json`, gated on `config.CONFIG.table_report_enabled` (`[refinement/agent.py#L108](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L108)`).
- `{data_profile}` injected into ablation, init-plan, and plan-refine instructions ([L160](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L160), [L204](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L204), [L261](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/refinement/agent.py#L261))

Author: Florian Hecht

## 12. New tuning stage (`sub_agents/tuning/`, improved-only)

**Goal:** a dedicated terminal hyperparameter stage after refinement — in-graph `choose`_* search → bake to fixed params → promote only if it beats the structural winner. Runs one leg per solution via `ParallelAgent`. (Plumbing/state keys live in `code_util.py`, see [§6](#6-tuning-stage-plumbing-improved-only).)


| Component                                                    | Pointer                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage agents + `ParallelAgent` assembly                      | `[tuning/agent.py#L418](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/tuning/agent.py#L418)`                                                                                                                                                                  |
| Promotion gate (bake must beat structural)                   | `[tuning/agent.py#L102](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/tuning/agent.py#L102)` — `promote_tuning_winner`                                                                                                                                        |
| Search / bake finish gates                                   | `[tuning/agent.py#L256](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/tuning/agent.py#L256)` `check_tune_implement_finish`, [L333](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/tuning/agent.py#L333) `check_tune_bake_finish` |
| Skip/fail markers + plan summary                             | `[tuning/agent.py#L34-L101](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/tuning/agent.py#L34-L101)`                                                                                                                                                          |
| Stage prompts (`TUNE_PLAN` / `TUNE_IMPLEMENT` / `TUNE_BAKE`) | `[tuning/prompt.py](agents/machine-learning-engineering/machine_learning_engineering/sub_agents/tuning/prompt.py)`                                                                                                                                                                          |


Author: Florian Hecht, Xiaomei Long

---

# Skill package, tasks & infra

## 13. Skrub DataOps skill package (improved-only)

**Goal:** a single on-demand ADK skill (`[skills/skrub-dataops-pipeline/](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/)`) the agents load at runtime (§2, §8) so they write DataOps-native code and avoid the concrete mistakes we hit in testing.

**Authorship:** Xiaomei Long contributed an initial draft of ~3–4 references; Florian Hecht wrote the rest and we kept his versions across the package (including reworks of the drafted ones). Content was distilled from the official skrub docs (notebooks, examples, API reference) and hardened with explicit "avoid this" rules, similar in spirit to the prompt guards, plus an iteratively grown common failure log + quick fixes collected while running the pipeline.

`[SKILL.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/SKILL.md)` is the router/policy that gets loaded first (before agent would decide to explore further references or not): DataOps-first rules, the default pipeline template, honest-holdout rules, per-stage load policy, and hard runtime constraints. The 14 references (`references/`):


| Reference                                                                                                                                                                | What it covers                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `[dataops_api_quickmap.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/dataops_api_quickmap.md)`           | Canonical DataOps pipeline shape (`skrub.var`/`X`/`y` → `.skb.apply`); load if unsure          |
| `[skrub_general_api.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/skrub_general_api.md)`                 | Non-DataOps helper API subset (`tabular_pipeline`, `TableVectorizer` args)                     |
| `[encoding_skrub.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/encoding_skrub.md)`                       | String/datetime encoders + `TableVectorizer`; encoder `choose`_* tuning                        |
| `[selectors_routing_skrub.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/selectors_routing_skrub.md)`     | `ApplyToCols`/`SelectCols`/`DropCols` selectors + split→concat routing                         |
| `[feature_engineering_skrub.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/feature_engineering_skrub.md)` | Derived features, ratios, cleaning, scaling, geo/datetime; TableReport-driven planner priority |
| `[joining_across_columns.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/joining_across_columns.md)`       | Multi-table groupby/merge inside DataOps expressions                                           |
| `[skrub_subsampling.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/skrub_subsampling.md)`                 | `.skb.subsample(...)` for fast iteration, full-data final eval                                 |
| `[choices_hparam_pattern.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/choices_hparam_pattern.md)`       | `choose`_*/`choose_from` + `make_randomized_search`/`make_grid_search`                         |
| `[ablation_dataops_template.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/ablation_dataops_template.md)` | Refinement ablation skeleton + stdout contract (load every ablation step)                      |
| `[tuning_dataops_template.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/tuning_dataops_template.md)`     | Terminal `tune_implement` skeleton; inject `choose`_* on focus block only (Xiaomei's draft)    |
| `[ensemble_dataops_patterns.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/ensemble_dataops_patterns.md)` | Ensemble merge Pattern A (multi-leg blend) / Pattern B (`VotingClassifier`)                    |
| `[submission_export.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/submission_export.md)`                 | Submission-only: full-train refit, `test_df` predict, `./final/submission.csv`                 |
| `[holdout_data_leakage.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/holdout_data_leakage.md)`           | Leakage-checker reference: incorrect vs correct bind/fit patterns                              |
| `[common_failure_fixes.md](agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references/common_failure_fixes.md)`           | Iteratively collected known skrub errors + quick fixes (debugging)                             |


Author: Florian Hecht

## 14. Tests

`[tests/](agents/machine-learning-engineering/tests/)` : `test_code_util.py`, `test_tuning_gate.py`, `test_ensemble_gate.py`, `test_leakage_gate.py`, `test_table_report_util.py`.

## 15. Evaluation harness & analysis

Base `[automated_evaluation/evaluate.py](automated_evaluation/evaluate.py)` written initially by Yuquan Cui, Florian Hecht appended to it and split out the orchestration/manifest concerns into separate scripts:

- `[run_experiments.py](automated_evaluation/run_experiments.py)` — batch orchestrator (multi-task/system/model runs, repeats, archiving, report generation).
- `[generate_tasks_manifest.py](automated_evaluation/generate_tasks_manifest.py)` — task manifest generation feeding the orchestrator.
- `[test-scripts/analyze_run.py](test-scripts/analyze_run.py)` — per-run metrics/analysis.

**Florian Hecht ran the experiments; a**rtifacts are under `[automated_evaluation/runs/](automated_evaluation/runs/)` and reports under `eval_results/`. Reproduction/analysis instructions: `[EXPERIMENTS.md](EXPERIMENTS.md)`, `[automated_evaluation/automated_evaluation.md](automated_evaluation/automated_evaluation.md)`.

## 16. Tasks, dependencies, docs, infra

- **Benchmark tasks:** collected and prepared the 10 task packs under `[tasks/](agents/machine-learning-engineering/machine_learning_engineering/tasks/)` (`abalone-regression`, `bike-sharing-regression`, `blueberry-yield-regression`, `covid19-forecasting-regression`, `employee-attrition-classification`, `introverts-extroverts-classification`, `multi-class-pred-obesity-risk`, `reservation-cancel-classification`, `restaurant-revenue-regression`, `spaceship-titanic`).
- **Dependencies:** small `[pyproject.toml](agents/machine-learning-engineering/pyproject.toml)` updates, bumped `google-adk` for agent skill compatibility (1.5 → 1.25) and added `skrub`, `ddgs` (the last part of the shared runtime-compat layer).
- **Vanilla setup:** `[sh-scripts/bootstrap_vanilla_baseline.sh](sh-scripts/bootstrap_vanilla_baseline.sh)` shell script for the vanilla bootstrap + `vanilla-baseline` worktree/branch setup.
- **Docs:** project docs under `[docs/](docs/)`.

---

## Attribution notes

- Authorship inferred and checked from `git log` across branches; first draft LLM-generated while managing necessary context, manually checked for correctness and refined.
- Line numbers refer to the current `main` state and may drift as files change; the function names are the stable anchor.

