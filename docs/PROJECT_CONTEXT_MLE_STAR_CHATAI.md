# Project Context: MLE-STAR + ChatAI Open-Weights Integration

## 1) Project Overview

- Repository base: `adk-samples` (Google ADK examples).
- Focused subproject: `python/agents/machine-learning-engineering/`.
- Agent system: MLE-STAR style multi-agent workflow:
  - `initialization_agent` -> `refinement_agent` -> `ensemble_agent` -> `submission_agent`
  - final callback writes `workspace/<task>/final_state.json`.
- Main objective: make MLE-STAR run reliably with **ChatAI open-weights models** (via OpenAI-compatible API / LiteLLM path), while preserving Gemini behavior.

## 2) Primary Goals

1. **Operational goal:** reliably reach `final_state.json` with ChatAI models.
2. **Integration goal:** keep Gemini path unchanged; only adjust behavior for non-Gemini/open-weights where needed.
3. **Main feature goal:** steer generated ML code to use **skrub DataOps** (not plain sklearn-only pipelines), starting with initialization stage and then expanding through the full workflow.

## 3) Environment and Runtime Notes

- `.env` uses OpenAI-compatible ChatAI endpoint for non-Gemini models (`openai/<model-id>`).
- `GOOGLE_GENAI_USE_VERTEXAI=0` during ChatAI runs.
- Known working baseline with Gemini: runs to completion and produces `final_state.json`.
- ChatAI models have mixed reliability (some immediate errors, some mid-run crashes, some rate-limited).

## 4) Main Issues Encountered

### Provider / tool-call compatibility
- Error:  
  `litellm.BadRequestError: OpenAIException - "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`
- Meaning: backend/model endpoint does not support ADK tool-call mode (`tool_choice=auto`) as configured.

### Response parsing fragility
- Error:  
  `TypeError: can only concatenate str (not "NoneType") to str`
- Root cause: response parts may contain `text=None` for some providers.
- Mitigation already applied: hardened `shared_libraries/common_util.py:get_text_from_response()` to ignore non-string text parts.

### Initialization ranking crash (secondary)
- Error in `rank_candidate_solutions`: `performance_results[0]` IndexError when no successful candidates.
- Interpretation: upstream candidate generation/evaluation failed; ranking lacked empty-list guard.

### ADK web graph/visualizer noise
- Seen occasional `agent_graph.py ... IndexError` traces; likely tooling/graph path issue, not necessarily core runtime logic.

### Rate limiting
- Frequent `429 API rate limit exceeded` on stronger models.
- Requires conservative loop/config settings and stable model selection.

## 5) Model Findings (from testing notes)

- Gemini models are most compatible with current baseline behavior.
- Some ChatAI models run far but may fail on rate limits or format/tool compatibility.
- Practical ChatAI candidates used during testing included:
  - `openai/llama-3.3-70b-instruct`
  - `openai/mistral-large-3-675b-instruct-2512`
  - `openai/qwen3.5-122b-a10b`

## 6) Current Code/Prompt Work Completed

### Completed code hardening
- `machine_learning_engineering/shared_libraries/common_util.py`
  - `get_text_from_response()` now safely handles empty/None text parts.

### Prompt engineering (initialization stage)
- File updated:  
  `machine_learning_engineering/sub_agents/initialization/prompt.py`
- Added concise but stronger constraints to:
  - prefer/require skrub DataOps patterns,
  - preserve skrub during debug/fix loops,
  - avoid plain sklearn-only fallback when solving initialization tasks.

## 7) Skrub Integration Direction

### Desired behavior
- Generated code should use **skrub DataOps abstractions** (e.g., `skrub.var`, `.skb.mark_as_X()`, `.skb.mark_as_y()`, `.skb.apply(...)`) as primary pipeline strategy.
- This should persist through:
  - initialization,
  - debug fixes,
  - merge/integration,
  - refinement and ensemble phases.

### Current status
- Observed progress: skrub/TableVectorizer/DataOps elements persisted further into flow (up to/near ablation in some runs).
- Remaining issue: model may still abandon skrub under error pressure.

## 8) Constraints for Any Future Changes

1. Do **not** break or alter Gemini behavior paths.
2. Keep prompt edits concise to avoid context bloat.
3. Prefer minimal, targeted hardening over broad rewrites.
4. Validate each phase incrementally:
   - init first, then refinement, then ensemble/submission.

## 9) Recommended Next Steps

1. Further tighten initialization prompts (short canonical skrub skeleton + stricter invalidity rules where needed).
2. Add protective guard/logging around initialization ranking when no candidates succeed.
3. Move to refinement prompts after initialization consistently produces skrub-based candidates.
4. Optional later phase: add ADK Skill-based skrub docs context for progressive disclosure.

## 10) Fast Session Restart Checklist

- Confirm env model:
  - `ROOT_AGENT_MODEL=openai/<chosen-model>`
  - ChatAI endpoint/key active
- Run from:
  - `python/agents/machine-learning-engineering/`
- Completion checks:
  - `workspace/california-housing-prices/final_state.json` exists
  - `submission_code_exec_result.returncode == 0`
- Inspect generated codes for skrub usage in init/merge/refine outputs.

