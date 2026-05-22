### How to run MLE-Star - Working Doc

```bash
cd /path/to/machine-learning-engineering
uv sync
uv run adk web # option 1: web UI
uv run adk run machine_learning_engineering # Option 2: CLI
```

Run these cmds too just to make sure all is setup:
```bash
uv add ddgs # IMPORTANT: DDG-search
uv add litellm
gcloud auth application-default login
```

Run these cmds if adk web process gets stall (e.g. not updating new .env)
```bash
unset ROOT_AGENT_MODEL OPENAI_API_BASE OPENAI_API_KEY
set -a
source .env
set +a
echo "$ROOT_AGENT_MODEL"
echo "$OPENAI_API_BASE"
uv run adk web
```

If any web process still running to kill:
```bash
# 1) Check if anything is still running
ps aux | rg "adk web|uv run.*adk web"
kill <PID>  # placeholder, use actual PID                                   
```

If .venv bleeds over from other repos (using wrong repo):
```bash
cd ~/uni/mle-star-skrub/mle-star_improved/agents/machine-learning-engineering

# hard-reset env influence
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME
hash -r

# rebuild this project's venv cleanly
rm -rf .venv
uv sync

# Verify: All paths must be correct repo
uv run python -c "import sys, machine_learning_engineering.agent as a, google.adk as g; print('python=', sys.executable); print('agent=', a.__file__); print('adk=', g.__file__)"

# Then restart
set -a; source .env; set +a
uv run adk web
```

Git problems:
```bash
# Get auth popup
GIT_ASKPASS= git -c core.askPass= -c credential.helper= push origin flo-dev
```

---

## Strategy A — Chat AI + OpenAI-compat web search

### Problem

MLE-STAR’s built-in **`google_search`** tool (`GoogleSearchTool`) only works with **native Gemini** APIs: it attaches Gemini-specific grounding config. Chat AI exposes **OpenAI-compatible** endpoints via LiteLLM (`OPENAI_API_BASE`, `ROOT_AGENT_MODEL` like `openai/…`). With that stack, calling `GoogleSearchTool` raises `ValueError: Google search tool is not supported for model openai/...`.

### What was added

1. **Compat search tool** — A normal **`FunctionTool`** whose implementation queries the public web with **DuckDuckGo** (`duckduckgo-search`, no API key). Exposed to the LLM as **`web_search(query, max_results)`** returning plain-text snippets plus URLs.

2. **Routing** — **`config.get_search_tools()`** selects tools the same way we choose LiteLLM vs native Gemini: if **`OPENAI_API_BASE` is set or `ROOT_AGENT_MODEL` starts with `openai/`**, use `[compat_web_search_tool]`; otherwise keep **`[google_search]`** for vanilla Gemini deployments.

### Files touched

| File | Role |
|------|------|
| `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/compat_web_search.py` | **`web_search`** + **`compat_web_search_tool = FunctionTool(web_search)`** |
| `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/config.py` | **`_uses_openai_compat_llm()`**, **`get_search_tools()`** |
| `agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/agent.py` | **`model_retriever_agent_*`**: `tools=config.get_search_tools()` |
| `agents/machine-learning-engineering/machine_learning_engineering/shared_libraries/debug_util.py` | **`debug_agent`**: same `tools=` swap |
| `agents/machine-learning-engineering/machine_learning_engineering/sub_agents/initialization/prompt.py` | **`MODEL_RETRIEVAL_INSTR`** — tells the agent to **`Call the web_search tool`** |
| `agents/machine-learning-engineering/pyproject.toml` | **`duckduckgo-search>=8.1.1`** dependency |
| `agents/machine-learning-engineering/tests/test_search_tools.py` | Routing tests (compat vs native) |
| `agents/machine-learning-engineering/.env.example` | Notes on DuckDuckGo vs Gemini search |

Related (earlier Chat AI wiring, not DDG-specific): **`get_agent_model()`** still returns **`LiteLlm`** when `openai/` or `OPENAI_API_BASE` + key are set (`config.py`). **`machine_learning_engineering/__init__.py`** loads **`.env`** via **`load_dotenv()`**.

### Code examples

**Routing (compact):**

```python
def _uses_openai_compat_llm() -> bool:
    model_name = os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001")
    api_base = os.environ.get("OPENAI_API_BASE")
    return bool(model_name.startswith("openai/") or api_base)


def get_search_tools() -> list[Any]:
    if _uses_openai_compat_llm():
        from machine_learning_engineering.shared_libraries.compat_web_search import (
            compat_web_search_tool,
        )
        return [compat_web_search_tool]
    from google.adk.tools.google_search_tool import google_search
    return [google_search]
```

(Location: `machine_learning_engineering/shared_libraries/config.py`.)

**DuckDuckGo tool (concept):**

```python
from google.adk.tools.function_tool import FunctionTool

def web_search(query: str, max_results: int = 8) -> str:
    from duckduckgo_search import DDGS
    ...
    with DDGS() as ddgs:
        for hit in ddgs.text(q, max_results=n):
            ...

compat_web_search_tool = FunctionTool(web_search)
```

(Location: `machine_learning_engineering/shared_libraries/compat_web_search.py`.)

**Agent wiring:**

```python
# model_retriever_agent_* (initialization) and debug_agent (debug_util)
tools=config.get_search_tools(),
```

### Env (Chat AI)

Typical `.env` entries kept working; compat search activates automatically when e.g.

- `OPENAI_API_BASE=https://chat-ai.academiccloud.de/v1`
- `OPENAI_API_KEY=...`
- `ROOT_AGENT_MODEL=openai/<model-name>`

### Caveats

- **Different from Google grounding** — snippets come from DuckDuckGo; quality varies.
- **Tool calling / `tool_choice`** — Some Chat AI backends (e.g. strict vLLM) reject OpenAI **`tool_choice: auto`** even for ADK **transfer/sub-agent** tools; that is separate from DDG search and may require a compliant model endpoint or provider configuration.
- **`tests/test_agents.py::test_happy_path`** depends on the chosen Chat AI model accepting tool-capable completions.
