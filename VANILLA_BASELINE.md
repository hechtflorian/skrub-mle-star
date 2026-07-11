# Vanilla MLE-STAR baseline

- **Baseline commit:** `ffa365cfd887cbff68667cbb3cbcaa52e07ffd39` (`ffa365c`)
- **Branch:** `vanilla-baseline` (worktree mode)
- **Prompts reverted to 6c96e03:** 0
- **skrub/optuna removed from pyproject:** 0
- **Created:** 2026-06-14T15:19:11+02:00

## What this includes

- OpenAI/ChatAI-compatible model routing (`ROOT_AGENT_MODEL`, LiteLLM)
- DuckDuckGo web search for non-Gemini models (`search_tool_util.py`)
- GPT-5 temperature compatibility (`get_compatible_temperature`)
- Safe response parsing (`common_util.get_text_from_response`)

## What this excludes (vs skrub-full on improve-refinement)

- ADK `skrub-dataops-pipeline` skill / SkillToolset
- TableReport profiling
- Terminal tuning stage (`sub_agents/tuning/`)

## Run

```bash
cd agents/machine-learning-engineering
# copy .env from mle-star_improved if needed
uv sync
uv run adk run machine_learning_engineering
```

Make sure your `.env` is correctly setup. For benchmarking vs. `skrub-full`, you can also check if both `shared_libraries/config.py` are the same.