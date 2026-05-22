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

