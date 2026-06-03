# MLE-STAR Initialization -> Refinement Map (Skrub Skill Integration)

This document maps the current agent flow for:
- `sub_agents/initialization`
- `sub_agents/refinement`

It shows:
1. Agent names defined in `agent.py`
2. Instruction source in `prompt.py`
3. Where Skrub skill tools are attached
4. Where prompts were modified for Skrub skill usage

---

## Quick flow overview

### Initialization stage flow
`initialization_agent`
-> `task_summarization_agent`
-> `init_parallel_agent`
-> `init_solution_gen_agent_{k}`
-> `model_retriever_loop_agent_{k}`
-> `model_retriever_agent_{k}`
-> `model_eval_and_debug_loop_agent_{k}_{m}`
-> `rank_agent_{k}`
-> `merge_and_debug_loop_agent_{k}_{i}` (for merge refs)
-> `selection_agent_{k}`
-> optional `check_data_use_and_debug_loop_agent_{k}`

### Refinement stage flow
`refinement_agent`
-> `ablation_and_refine_loop_agent_{k}`
-> `ablation_and_refine_agent_{k}`
-> `ablation_and_debug_loop_agent_{k}`
-> `ablation_summary_agent_{k}`
-> `init_plan_loop_agent_{k}`
-> `init_plan_implement_agent_{k}`
-> `refine_inner_loop_agent_{k}`
-> `plan_refine_and_implement_agent_{k}`
-> `plan_refine_agent_{k}`
-> `plan_implement_agent_{k}`

`k` = solution index, `m` = model-candidate index, `i` = merge index.

---

## Initialization: agent-by-agent mapping

| Agent name in `initialization/agent.py` | Instruction source | Skrub skill tools | Notes |
|---|---|---|---|
| `task_summarization_agent` | `prompt.SUMMARIZATION_AGENT_INSTR` | No | Summarization only. |
| `model_retriever_agent_{k}` | `get_model_retriever_agent_instruction()` -> `prompt.MODEL_RETRIEVAL_INSTR` | Yes (`skill + search`) | Modified for skill-loading guidance. |
| `model_eval_and_debug_loop_agent_{k}_{m}` | `get_model_eval_agent_instruction()` -> `prompt.MODEL_EVAL_INSTR` (run phase) + `debug_prompt.BUG_REFINE_INSTR` (debug phase) | Yes (`run`: skill-only, `debug`: skill+search) | Core code generation path. |
| `rank_agent_{k}` | No LLM instruction (callback-only) | No | Ranking state update only. |
| `merge_and_debug_loop_agent_{k}_{i}` | `get_merger_agent_instruction()` -> `prompt.CODE_INTEGRATION_INSTR` (run phase) + `debug_prompt.BUG_REFINE_INSTR` (debug phase) | Yes (`run`: skill-only, `debug`: skill+search) | Integration code path. |
| `selection_agent_{k}` | No LLM instruction (callback-only) | No | Selection state update only. |
| `check_data_use_and_debug_loop_agent_{k}` (optional) | `get_check_data_use_instruction()` -> `prompt.CHECK_DATA_USE_INSTR` (run phase) + `debug_prompt.BUG_REFINE_INSTR` (debug phase) | Yes (`run`: skill-only, `debug`: skill+search) | Optional data-use refinement path. |

---

## Refinement: agent-by-agent mapping

| Agent name in `refinement/agent.py` | Instruction source | Skrub skill tools | Notes |
|---|---|---|---|
| `ablation_agent_{k}` | `get_ablation_agent_instruction()` -> `prompt.ABLATION_INSTR` or `prompt.ABLATION_SEQ_INSTR` | Yes (skill-only) | Code-producing ablation stage. |
| `ablation_summary_agent_{k}` | `get_ablation_summary_agent_instruction()` -> `prompt.SUMMARIZE_ABLATION_INSTR` | No | Summary only. |
| `init_plan_agent_{k}` | `get_init_plan_agent_instruction()` -> `prompt.EXTRACT_BLOCK_AND_PLAN_INSTR` or `prompt.EXTRACT_BLOCK_AND_PLAN_SEQ_INSTR` | Yes (skill-only) | Planning + block extraction. |
| `init_plan_implement_agent_{k}` | `get_plan_implement_agent_instruction()` -> `prompt.IMPLEMENT_PLAN_INSTR` (run phase) + `debug_prompt.BUG_REFINE_INSTR` (debug phase) | Yes (`run`: skill-only, `debug`: skill+search) | Initial plan implementation code path. |
| `plan_refine_agent_{k}` | `get_plan_refinement_instruction()` -> `prompt.PLAN_REFINEMENT_INSTR` | Yes (skill-only) | Plan refinement stage. |
| `plan_implement_agent_{k}` | `get_plan_implement_agent_instruction()` -> `prompt.IMPLEMENT_PLAN_INSTR` (run phase) + `debug_prompt.BUG_REFINE_INSTR` (debug phase) | Yes (`run`: skill-only, `debug`: skill+search) | Iterative implementation code path. |

---

## Shared debug utility mapping

In `shared_libraries/debug_util.py`:

- `get_debug_inner_loop_agent(...)`
  - internal `debug_agent` uses:
    - instruction: `debug_prompt.BUG_REFINE_INSTR`
    - tools: `skill_tool_util.get_skill_and_search_tools(...)` (skill + search)

- `get_run_and_debug_agent(...)`
  - internal `run_agent` uses:
    - instruction passed via `instruction_func` from calling sub-agent
    - tools passed via `tools` arg (default `[]`)
  - This is how initialization/refinement run phases get skill-only tools where needed.

---

## Prompt modifications for Skrub (current)

### Initialization prompts (`sub_agents/initialization/prompt.py`)
Skrub-focused edits are present in:
- `MODEL_RETRIEVAL_INSTR`
- `MODEL_EVAL_INSTR`
- `BUG_REFINE_INSTR`
- `CODE_INTEGRATION_INSTR`
- `CHECK_DATA_USE_INSTR`

### Refinement prompts (`sub_agents/refinement/prompt.py`)
Skrub-focused edits are present in:
- `ABLATION_INSTR`
- `ABLATION_SEQ_INSTR`
- `EXTRACT_BLOCK_AND_PLAN_INSTR`
- `EXTRACT_BLOCK_AND_PLAN_SEQ_INSTR`
- `PLAN_REFINEMENT_INSTR`
- `IMPLEMENT_PLAN_INSTR`

### Shared debug prompt (`shared_libraries/debug_prompt.py`)
Skrub-focused edits are present in:
- `BUG_REFINE_INSTR`

---

## Notes on scope control (anti-bloat)

- Agents without substantive code generation or DataOps planning are left without skill tools.
- Callback-only or summary-only agents (ranking, selection, ablation summary) are intentionally not skill-enabled.
- Search tools were not broadly added to refinement run agents and search is kept alligning with vanilla MLE-Stars search capabilites; refinement run paths use skill-only where explicitly wired.

