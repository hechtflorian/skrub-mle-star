# MLE-STAR Agent Deep Dive (`machine_learning_engineering/`)

This document explains how the MLE-STAR sample in `machine_learning_engineering/` works internally, with emphasis on:

- `sub_agents/`
- `shared_libraries/`

It is written from the actual code structure in this repository.

---

## 1) High-Level Architecture

The runtime pipeline is defined in `machine_learning_engineering/agent.py`:

1. `initialization_agent`
2. `refinement_agent`
3. `ensemble_agent`
4. `submission_agent`

These run in a `SequentialAgent` called `mle_pipeline_agent`.

The entry `root_agent` is a frontdoor `Agent` (`name="mle_frontdoor_agent"`) that delegates to this pipeline.  
After the pipeline finishes, `save_state()` writes `final_state.json` under:

- `./machine_learning_engineering/workspace/<task_name>/final_state.json`

---

## 2) Environment and Bootstrapping

`machine_learning_engineering/__init__.py` sets defaults:

- `GOOGLE_CLOUD_PROJECT` from ADC (`google.auth.default()`)
- `GOOGLE_CLOUD_LOCATION=global`
- `GOOGLE_GENAI_USE_VERTEXAI=True` (only if not already set)

Important behavior:

- `setdefault()` means your shell/.env value wins if already set.
- The model used by all major agents comes from `ROOT_AGENT_MODEL` via `shared_libraries/config.py` and `agent.py`.

---

## 3) Config and Global State (`shared_libraries/config.py`)

`DefaultConfig` is copied into ADK state at task start (in initialization `prepare_task`).

Key parameters:

- Data/task:
  - `data_dir` (default `./machine_learning_engineering/tasks/`)
  - `task_name` (default `california-housing-prices`)
  - `workspace_dir` (default `./machine_learning_engineering/workspace/`)
- Model:
  - `agent_model` from `ROOT_AGENT_MODEL` (fallback `gemini-2.0-flash-001`)
- Control loops:
  - `max_retry`, `max_debug_round`, `max_rollback_round`
  - `inner_loop_round`, `outer_loop_round`, `ensemble_loop_round`
- Safety toggles:
  - `use_data_leakage_checker`
  - `use_data_usage_checker`

This config controls how expensive and persistent the multi-agent search/debug process is.

---

## 4) Search Tool Routing (`shared_libraries/search_tool_util.py`)

This module solves the Gemini vs non-Gemini search-tool compatibility:

- If model is Gemini (`is_gemini_model(model_name)`): use ADK `google_search`.
- Otherwise: use a custom DuckDuckGo function tool (`ddg_web_search`) via `duckduckgo_search`.

So:

- `ROOT_AGENT_MODEL=gemini-*` -> native `google_search`
- `ROOT_AGENT_MODEL=openai/...` or other non-Gemini -> DDG tool

This routing is used in both:

- initialization model retrieval
- debug agents

---

## 5) Shared Execution Semantics (`shared_libraries/code_util.py`)

This module is central. It standardizes how generated code is executed and scored.

### 5.1 Code execution

`run_python_code(...)`:

- writes code to a file in the current run directory
- executes `python <file>`
- captures `stdout`, `stderr`, `returncode`, `execution_time`

### 5.2 Scoring contract

`extract_performance_from_text(...)` expects this exact marker in script output:

- `Final Validation Performance: <float>`

If parsing fails or script fails:

- score becomes fallback:
  - `1e9` when lower is better
  - `0` when higher is better

### 5.3 State-key naming conventions

`get_code_state_key()` and `get_code_execution_result_state_key()` map agent type -> canonical state keys, for example:

- `model_eval` -> `init_code_*`, `init_code_exec_result_*`
- `plan_implement` -> `train_code_improve_*`
- `ensemble_plan_implement` -> `ensemble_code_*`
- `submission` -> `submission_code`, `submission_code_exec_result`

### 5.4 When code is actually executed

`get_run_code_condition(...)` imposes guardrails:

- Most stages require no `exit()`
- General train/eval stages require the output marker logic in code
- Submission stage must involve `submission.csv`

This is why prompt formatting matters a lot.

---

## 6) Debug Framework (`shared_libraries/debug_util.py`)

This module creates reusable "run + debug" pipelines.

## 6.1 Core pattern

`get_run_and_debug_agent(...)` builds a nested structure:

1. `run_agent` (generate/modify code)
2. optional data-leakage checker sequence
3. run loop (`max_retry`)
4. debug inner loop:
   - summarize bug
   - debug code with bug context and (model-compatible) web search
5. rollback loop (`max_rollback_round`)

### 6.2 Key callbacks

- `get_code_from_response(...)`:
  - extracts code from model output
  - updates the right state key
  - triggers `code_util.evaluate_code()`
- `check_bug_existence(...)`:
  - skips debug when returncode already success
- `check_rollback(...)`:
  - clears failed execution result state for retry on next rollback

### 6.3 Special handling

- For `plan_implement`, non-debug path applies code-block replacement into previous full script.
- For `check_data_use`, it can stop once output says all info is used.

---

## 7) Data Leakage Checker (`shared_libraries/check_leakage_util.py`)

Optional module (enabled by `use_data_leakage_checker=True`).

Flow:

1. Ask LLM to extract leakage-prone block and verdict (`Yes Data Leakage` / `No Data Leakage`)
2. Loop until extraction is valid (`max_retry`)
3. If leakage found, ask LLM to rewrite the leakage block
4. Replace block into current code and re-evaluate immediately

It uses strict JSON parsing (`parse_leakage_status`) and state flags:

- `*_extract_status_*`
- `*_leakage_status_*`
- `*_leakage_block_*`
- `*_skip_data_leakage_check_*`

---

## 8) Utility Helpers (`shared_libraries/common_util.py`)

Small but important utilities:

- `get_text_from_response(...)`: concatenates text parts from LLM response
- `set_random_seed(...)`: sets Python/NumPy/Torch determinism flags
- `copy_file(...)`: safe file copy helper

Note: `get_text_from_response` assumes `.text` parts are concatenable strings. If a `.text` part is `None`, runtime errors can occur unless handled upstream.

---

## 9) Sub-Agent: Initialization (`sub_agents/initialization/agent.py`)

Goal: produce initial candidate pipelines for each solution stream.

### 9.1 What it does

For each solution `k` (`1..num_solutions`):

1. Summarize task
2. Retrieve model candidates (with search tool)
3. For each candidate model:
   - generate code (`model_eval`)
   - run/debug to get executable baseline
4. Rank candidates by score
5. Iteratively merge top with others (`merger`) + run/debug
6. Select best merged solution
7. Optionally check data usage

All solution branches run in parallel via `init_parallel_agent`.

### 9.2 Workspace setup

Each branch creates:

- `workspace/<task_name>/<task_id>/`
  - `input/` copy of task files
  - `model_candidates/`

It intentionally excludes files with `"answer"` in filename.

### 9.3 Important state artifacts

- `init_<task>_model_<i>`
- `init_code_exec_result_<task>_<model>`
- `performance_results_<task>`
- `train_code_0_<task>`
- `train_code_exec_result_0_<task>`

This provides seed solutions for refinement.

---

## 10) Sub-Agent: Refinement (`sub_agents/refinement/agent.py`)

Goal: improve each initial solution through ablation-guided targeted edits.

Each solution branch runs in parallel. Inside each branch:

1. Ablation generation + debug
2. Summarize ablation results
3. Extract code block + initial improvement plan
4. Implement plan (run/debug)
5. Inner loop: refine plan and re-implement
6. Outer-loop update selects best improved code

### 10.1 Core idea

Instead of rewriting whole script every time, refinement often:

- identifies a high-impact block
- edits that block
- reinserts into existing script

### 10.2 Loop controls

- `outer_loop_round` controls how many ablation->refine rounds
- `inner_loop_round` controls plan refinement attempts each round
- standard debug/retry limits still apply through `debug_util`

### 10.3 Key state

- `refine_step_<task>`
- `refine_code_block_<step>_<task>`
- `refine_plans_<step>_<task>`
- `train_code_improve_*`
- `train_code_<step+1>_<task>`

`update_outer_loop_states()` compares improvements by score and persists the best next-step script.

---

## 11) Sub-Agent: Ensemble (`sub_agents/ensemble/agent.py`)

Goal: combine refined solutions into one stronger script.

Flow:

1. Create ensemble workspace at `workspace/<task_name>/ensemble/`
2. Build initial ensemble plan from all refined solution codes
3. Implement plan via run/debug
4. Iteratively refine plan and re-implement (`ensemble_loop_round`)

Outputs are tracked as:

- `ensemble_plans`
- `ensemble_code_<iter>`
- `ensemble_code_exec_result_<iter>`

Plan refinement uses previous ensemble scores to bias toward better plans.

---

## 12) Sub-Agent: Submission (`sub_agents/submission/agent.py`)

Goal: produce final executable script that generates submission file.

Flow:

1. Select best code among:
   - refined per-solution outputs
   - ensemble outputs
2. Prompt agent to minimally add test-time inference + CSV write
3. Run/debug until `submission_code_exec_result` exists

Expected artifact:

- `./final/submission.csv` under ensemble workspace

The submission agent is also built using the same generic `debug_util.get_run_and_debug_agent(...)`.

---

## 13) Prompt Design Strategy

Prompt files are split by stage:

- `sub_agents/initialization/prompt.py`
- `sub_agents/refinement/prompt.py`
- `sub_agents/ensemble/prompt.py`
- `sub_agents/submission/prompt.py`
- `shared_libraries/debug_prompt.py`
- `shared_libraries/data_leakage_prompt.py`

Common constraints across prompts:

- return only code block when needed
- avoid `exit()`
- always print `Final Validation Performance: ...` (except ablation where output format differs)
- use local `./input` data only

This strict format is required by `code_util` parsing and loop control.

---

## 14) End-to-End State and Artifact Trace

Typical artifact path for default task:

- `machine_learning_engineering/workspace/california-housing-prices/`
  - `1/`, `2/` solution branches (initialization/refinement artifacts)
  - `ensemble/` (ensemble + submission artifacts)
  - `final_state.json` (if full pipeline finishes)

If `final_state.json` exists, pipeline callback `save_state` has run at the end of `mle_pipeline_agent`.

---

## 15) Practical Extension Points

If you want to customize behavior, these are highest leverage:

1. `shared_libraries/config.py`
   - reduce/increase loop budgets
   - set safety toggles
2. `shared_libraries/search_tool_util.py`
   - swap or add external search backends
3. `shared_libraries/debug_util.py`
   - tune run/debug/rollback policy
4. stage prompt files
   - control coding style, dependency usage, output format

---

## 16) Common Failure Modes (from code behavior)

1. **Score parsing fails**
   - Generated code did not print `Final Validation Performance: ...`
   - Result: fallback worst score.

2. **No code execution**
   - `get_run_code_condition` not satisfied (e.g., includes `exit()`, missing expected markers).

3. **Tool mismatch**
   - Wrong model prefix can force incompatible search tool path.

4. **State parse fragility**
   - Several places parse JSON-like outputs from free-form model text.
   - Invalid format can cause retries/empty results.

5. **Long loop runtimes**
   - Combined loops (`max_retry`, debug rounds, rollback, outer/inner/ensemble loops) multiply quickly.

---

## 17) Mental Model in One Sentence

MLE-STAR here is a stateful ADK orchestration that repeatedly **generate -> execute -> score -> debug -> refine -> ensemble -> finalize**, with most behavior controlled by prompt contracts and shared loop/execution utilities.

